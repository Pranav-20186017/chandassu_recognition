import json
import tarfile
from dataclasses import replace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import T5Config

from chandassu import LABELS
from chandassu.artifacts import load_package
from chandassu.config import TrainConfig, resolve
from chandassu.data import Corpus, Poem, assert_isolated, normalize_text, plain_folds, poem_dict, work_groups
from chandassu.evaluation.metrics import fit_temperature, line_poem_metrics, softmax
from chandassu.inference.server import create_app
from chandassu.io import atomic_json, sha256
from chandassu.models.encoding import vocabulary_for
from chandassu.models.factory import build_model
from chandassu.training.engine import train_fit
from chandassu.training.optimization import backward_window


def poems(role, works=1):
    return [
        Poem(
            f"{role}-{w}-{i}",
            f"{role}-work-{w}",
            "author",
            tuple(f"{role} {chr(97 + w)} {chr(97 + i)} {chr(97 + j)}" for j in range(4)),
            label,
        )
        for w in range(works)
        for i, label in enumerate(LABELS)
    ]


def tiny_model(settings, vocabulary):
    encoder = T5Config(vocab_size=384, d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8, dropout_rate=0.1)
    return build_model(settings, vocabulary, encoder_config=encoder.to_dict(), pretrained=False)


def config(arm):
    return TrainConfig(
        model_type=arm,
        epochs=3,
        minimum_epochs=1,
        patience=10,
        microbatch_poems=3,
        effective_batch_poems=6,
        evaluation_batch_poems=4,
        width=8,
        max_tokens=128,
        device="cpu",
        precision="fp32",
        cpu_threads=1,
        shuffle_poems=True,
        split_manifest=None,
    )


def test_expanded_corpus_preserves_baseline_roles_and_frozen_tests():
    old, expanded = Corpus("data/v1"), Corpus("data/v2")
    original = old.final_parts("splits/final_h100.json", 0.15, 142)
    parts = expanded.final_parts("splits/train_expansion.json", 0.15, 142)

    def ids(rows):
        return {p.poem_id for p in rows}

    assert ids(original["fit"]) <= ids(parts["fit"])
    for role in ("selection", "calibration"):
        assert parts[role] == original[role]
    assert (expanded.folder / "validation.jsonl").read_bytes() == (old.folder / "validation.jsonl").read_bytes()
    frozen = json.loads((expanded.folder / "EXPANSION.json").read_text())["allocation"]["frozen_new_work_holdout"]
    held = set(frozen["holdout_poem_ids"])
    test = expanded.poems("test")
    assert ids(test) == ids(old.poems("test")) | held
    assert {p.source for p in test if p.poem_id in held} == set(frozen["holdout_works"])
    assert len(parts["fit"]) == 7575 and len(test) == 2953
    assert_isolated({**parts, "test": test})
    rows = [p for role in ("train", "validation", "test") for p in expanded.poems(role)]
    assert len(rows) == len(ids(rows)) == 13295
    new = [p for p in rows if p.metadata.get("collection") == "expansion_20261011"]
    assert len(new) == 1777 and len({p.source for p in new}) == 34
    assert len(ids(parts["fit"]) - ids(original["fit"])) == 1058
    lookup = {p.poem_id: p for p in rows}
    for role in ("train", "validation", "test"):
        for poem in old.poems(role):
            retained = lookup[poem.poem_id]
            assert (retained.lines, retained.label, retained.source, retained.author) == (
                poem.lines,
                poem.label,
                poem.source,
                poem.author,
            )
    cohorts = expanded.manifest["test_cohorts"]
    assert set(cohorts["v1_test"]["ids"]) == ids(old.poems("test"))
    assert set(cohorts["v2_new_work_holdout"]["ids"]) == held
    preset = resolve(["--model_type=cnn", "--config=configs/cnn.json"])
    assert preset.data_dir == "data/v2" and preset.init_checkpoint is None
    assert preset.split_manifest == "splits/train_expansion.json"
    assert resolve(["--model_type=byt5", "--config=configs/byt5.json"]).data_dir == "data/v1"


def test_entrypoint_checks_test_isolation_before_training(tmp_path, monkeypatch):
    import train

    class Dataset:
        def __init__(self, _folder):
            pass

        def final_parts(self, *_args):
            return {"fit": poems("fit"), "selection": poems("selection"), "calibration": poems("calibration")}

        def poems(self, _role):
            return poems("fit")

    monkeypatch.setattr(train, "Corpus", Dataset)

    def forbidden_training(*_args):
        pytest.fail("Training must not start with a leaked test work")

    monkeypatch.setattr(train, "train_fit", forbidden_training)
    with pytest.raises(ValueError, match="leakage/duplicate"):
        train.main(["--model_type=cnn", "--output_dir", str(tmp_path)])


def test_evaluation_reports_both_fixed_test_cohorts(tmp_path, monkeypatch):
    from scripts import evaluate

    corpus = Corpus("data/v2")
    test = corpus.poems("test")

    class Model:
        device = "cpu"
        config = {"temperature": 1.0}
        metadata = {"partitions": {"fit": {"poem_ids": []}}}

        def __init__(self, _folder):
            pass

        def info(self):
            return {"device": self.device}

        def logits(self, lines, _batch_size):
            assert len(lines) == 4 * len(test)
            result = np.zeros((len(lines), 4))
            for i, poem in enumerate(test):
                result[4 * i : 4 * i + 4, LABELS.index(poem.label)] = 10
            return result

    monkeypatch.setattr(evaluate, "Predictor", Model)
    output = tmp_path / "test.json"
    evaluate.main(["--model_dir", str(tmp_path), "--data_dir=data/v2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["metrics"]["poem"]["observations"] == 2953
    assert report["cohort_metrics"]["v1_test"]["poem"]["observations"] == 2234
    assert report["cohort_metrics"]["v2_new_work_holdout"]["poem"]["observations"] == 719
    Model.metadata = {"partitions": {"fit": {"poem_ids": [test[0].poem_id]}}}
    with pytest.raises(ValueError, match="overlap model development"):
        evaluate.main(["--model_dir", str(tmp_path), "--data_dir=data/v2", "--output", str(output)])


def test_fixed_cnn_replicates_select_before_test_resume_and_package_all(tmp_path, monkeypatch):
    import train
    from chandassu.evaluation.metrics import classification
    from chandassu.inference.collection import ModelCollection
    from scripts import evaluate

    data = tmp_path / "data"
    data.mkdir()
    parts = {"fit": poems("fit", 2), "selection": poems("selection"), "calibration": poems("calibration")}
    test = poems("test")
    for role, rows in {
        "train": parts["fit"] + parts["calibration"],
        "validation": parts["selection"],
        "test": test,
    }.items():
        (data / f"{role}.jsonl").write_text("".join(json.dumps(poem_dict(p)) + "\n" for p in rows))
    atomic_json(
        data / "manifest.json",
        {
            "schema_version": 2,
            "labels": list(LABELS),
            "files": {name: sha256(data / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl")},
        },
    )
    corpus = Corpus(data)
    output = tmp_path / "replicates"
    settings = replace(config("cnn"), mode="replicates", split_manifest="fixed.json", data_dir=str(data))
    actual_fit = train.train_fit
    actual_evaluate = evaluate.main
    fitted = []

    def checked_fit(child, rows, folder):
        assert rows == parts
        assert child.learning_rate == settings.learning_rate and child.init_checkpoint is None
        fitted.append(child.seed)
        return actual_fit(child, rows, folder)

    def checked_evaluate(args):
        assert fitted[-3:] == [17, 42, 73]
        selection = json.loads((output / "selection.json").read_text())
        assert selection["test_used_for_selection"] is False
        assert len(selection["members"]) == 3
        return actual_evaluate(args)

    monkeypatch.setattr(train, "train_fit", checked_fit)
    monkeypatch.setattr(evaluate, "main", checked_evaluate)
    selection = train.train_replicates(settings, corpus, parts, output)
    assert fitted == [17, 42, 73]
    assert selection["best_single"] == min(
        selection["members"], key=lambda m: (-m["validation_poem_macro_f1"], m["validation_line_ce"], m["seed"])
    )
    assert selection["huggingface_candidate"] == selection["best_single"]["model_dir"]
    folders = [output / f"seed-{seed}/model" for seed in (17, 42, 73)]
    before = [sha256(folder / "model.safetensors") for folder in folders]
    for seed in (17, 42, 73):
        result = json.loads((output / f"seed-{seed}/result.json").read_text())
        assert result["calibration"]["poems"] == len(parts["calibration"])
        assert (output / f"seed-{seed}/test.json").is_file()
    # The evaluation ensemble must match the existing local inference ensemble.
    collection = ModelCollection(folders, strategy="ensemble")
    predicted = [collection.predict("\n".join(p.lines)) for p in test]
    probabilities = np.array([[c["probability"] for c in p["classes"]] for p in predicted])
    expected = classification(np.array([LABELS.index(p.label) for p in test]), probabilities)
    report = json.loads((output / "ensemble-test.json").read_text())
    assert report["metrics"]["poem"]["confusion_matrix"] == expected["confusion_matrix"]
    assert report["metrics"]["poem"]["cross_entropy"] == pytest.approx(expected["cross_entropy"])
    with tarfile.open(tmp_path / "replicates.tar.gz") as archive:
        names = archive.getnames()
        assert "runs/replicates/selection.json" in names and "data/data/manifest.json" in names
        for seed in (17, 42, 73):
            assert f"runs/replicates/seed-{seed}/model/model.safetensors" in names
            assert f"runs/replicates/seed-{seed}/best.pt" in names
            assert f"runs/replicates/seed-{seed}/last.pt" in names
    # Completed seeds are verified on resume without further optimizer updates.
    import chandassu.training.engine as engine

    monkeypatch.setattr(engine, "backward_window", lambda *_a, **_k: pytest.fail("Completed replica retrained"))
    assert train.train_replicates(replace(settings, resume=True), corpus, parts, output) == selection
    assert [sha256(folder / "model.safetensors") for folder in folders] == before
    with pytest.raises(ValueError, match="plan changed"):
        train.train_replicates(replace(settings, resume=True, learning_rate=0.0003), corpus, parts, output)


def test_replicate_arguments_reject_repeated_seeds_byt5_and_initialization():
    args = ["--model_type=cnn", "--config=configs/cnn.json", "--mode=replicates"]
    assert resolve(args).seeds == (17, 42, 73)
    assert resolve(args + ["--seeds", "42", "73"]).seeds == [42, 73]
    with pytest.raises(ValueError, match="distinct integer seeds"):
        resolve(args + ["--seeds", "17", "17"])
    with pytest.raises(ValueError, match="fresh initialization"):
        resolve(args + ["--init_checkpoint=old-model"])
    with pytest.raises(ValueError, match="require CNN"):
        resolve(["--model_type=byt5", "--mode=replicates"])


@pytest.mark.parametrize("arm", ["cnn", "byt5"])
def test_interrupted_resume_matches_uninterrupted(arm, tmp_path, monkeypatch):
    import chandassu.training.engine as engine

    parts = {"fit": poems("fit", 2), "selection": poems("validation"), "calibration": poems("calibration")}
    settings = config(arm)
    baseline = train_fit(settings, parts, tmp_path / "baseline", model_factory=tiny_model)
    actual_save = engine.atomic_torch

    def interrupt_after_epoch(path, value):
        actual_save(path, value)
        if path.name == "last.pt" and value["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(engine, "atomic_torch", interrupt_after_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_fit(settings, parts, tmp_path / "interrupted", model_factory=tiny_model)
    monkeypatch.setattr(engine, "atomic_torch", actual_save)
    resumed = train_fit(replace(settings, resume=True), parts, tmp_path / "interrupted", model_factory=tiny_model)
    a, _, _, _ = load_package(tmp_path / "baseline/model")
    b, _, _, _ = load_package(tmp_path / "interrupted/model")
    assert all(torch.equal(a.state_dict()[key], value) for key, value in b.state_dict().items())
    assert resumed["selection_metrics"] == baseline["selection_metrics"]
    assert resumed["calibration"] == baseline["calibration"]
    assert resumed["selected_epoch"] == baseline["selected_epoch"]
    # All completed epochs, including the partial last effective batch, match.
    x = torch.load(tmp_path / "baseline/last.pt", weights_only=False)
    y = torch.load(tmp_path / "interrupted/last.pt", weights_only=False)
    assert all(torch.equal(x["model"][key], value) for key, value in y["model"].items())
    assert torch.equal(x["rng"]["cpu"], y["rng"]["cpu"])
    assert (
        train_fit(replace(settings, resume=True), parts, tmp_path / "interrupted", model_factory=tiny_model) == resumed
    )
    with pytest.raises(ValueError, match="changed"):
        train_fit(replace(settings, resume=True, seed=42), parts, tmp_path / "interrupted", model_factory=tiny_model)


def test_weighted_accumulation_matches_full_update_including_partial_window():
    a = torch.nn.Linear(5, 4)
    b = torch.nn.Linear(5, 4)
    b.load_state_dict(a.state_dict())
    x = torch.arange(35, dtype=torch.float32).reshape(7, 5) / 20
    y = torch.tensor([0, 0, 0, 1, 2, 2, 3])
    weights = torch.tensor([0.3, 1.0, 0.7, 2.0])
    import contextlib

    backward_window(
        a, [(x[:3], y[:3]), (x[3:6], y[3:6]), (x[6:], y[6:])], weights, weights, "cpu", contextlib.nullcontext
    )
    torch.nn.functional.cross_entropy(b(x), y, weight=weights).backward()
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("arm", ["cnn", "byt5"])
def test_padding_does_not_change_prediction(arm):
    settings = config(arm)
    vocabulary = vocabulary_for(arm, 128, poems("fit"))
    model = tiny_model(vars(settings), vocabulary).eval()
    x = torch.tensor([vocabulary.encode("fit 1")])
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(torch.nn.functional.pad(x, (0, 17))), atol=1e-5, rtol=1e-5)


def test_prepared_corpus_is_complete_and_frozen():
    corpus = Corpus("data/v1")
    parts = corpus.final_parts("splits/final_h100.json", 0.15, 142)
    assert [len(parts[k]) for k in ("fit", "selection", "calibration")] == [6517, 1859, 908]
    assert len(corpus.poems("test")) == 2234
    works = json.loads((corpus.folder / "works.json").read_text())
    assert sum(w["target_poems"] for w in works) == 11518
    assert sum(w["target_poems"] == 0 for w in works) == 10
    assert_isolated({**parts, "historical": corpus.poems("test")})
    vocabulary = vocabulary_for("cnn", 256, parts["fit"])
    assert vocabulary.tokens == {
        c: i + 2 for i, c in enumerate(sorted({c for p in parts["fit"] for t in p.lines for c in t}))
    }
    assert normalize_text("కా\u200c\u200d\ufeff  రా") == "కా\u200c\u200d రా"


def test_work_and_edition_groups_stay_together():
    data = poems("group", 6)
    data[4] = replace(data[4], metadata={"edition_family": "same"})
    data[8] = replace(data[8], metadata={"edition_family": "same"})
    groups = work_groups(data)
    assert groups[4] == groups[8]
    folds = plain_folds(data, 3, 142)
    assert len(folds) == 3
    held_out = [p.poem_id for fold in folds for p in fold["selection"]]
    assert sorted(held_out) == sorted(p.poem_id for p in data)
    for fold in folds:
        assert_isolated(fold)


@pytest.mark.parametrize("violation", ["work", "edition", "layout", "id"])
def test_leakage_is_rejected(violation):
    fit, selection = poems("fit"), poems("selection")
    if violation == "work":
        selection[0] = replace(selection[0], source=fit[0].source)
    if violation == "edition":
        fit[0] = replace(fit[0], metadata={"edition_family": "same"})
        selection[0] = replace(selection[0], metadata={"edition_family": "same"})
    if violation == "layout":
        selection[0] = replace(selection[0], lines=tuple(t + " !" for t in fit[0].lines))
    if violation == "id":
        selection[0] = replace(selection[0], poem_id=fit[0].poem_id)
    with pytest.raises(ValueError, match="leakage|duplicate"):
        assert_isolated({"fit": fit, "selection": selection})


def test_cli_is_flat_and_explicit_flags_win(tmp_path):
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"seed": 42, "learning_rate": 3e-4, "epochs": 40}))
    c = resolve(["--model_type=cnn", "--config", str(preset), "--seed=73", "--microbatch_poems=8"])
    assert (c.seed, c.learning_rate, c.epochs, c.microbatch_poems) == (73, 3e-4, 40, 8)
    assert resolve(["--model_type=byt5"]).learning_rate == 3e-5
    with pytest.raises(ValueError, match="Plain CV"):
        resolve(["--model_type=cnn", "--mode=cross_validate"])
    preset.write_text('{"cnn":{"seed":42}}')
    with pytest.raises(ValueError, match="flat"):
        resolve(["--model_type=cnn", "--config", str(preset)])
    preset.write_text('{"learning_rate":NaN}')
    with pytest.raises(ValueError, match="finite"):
        resolve(["--model_type=cnn", "--config", str(preset)])


def test_poem_aggregation_and_calibration():
    logits = np.array([[4.0, 1, 0, 0], [1.0, 4, 0, 0], [4.0, 1, 0, 0], [4.0, 1, 0, 0]] * 4)
    y = np.repeat(np.arange(4), 4)
    result = fit_temperature(logits, y)
    assert result["after_nll"] <= result["before_nll"]
    expected = softmax(logits).reshape(-1, 4, 4).mean(1)
    metrics = line_poem_metrics(logits, y)
    assert metrics["poem"]["accuracy"] == float((expected.argmax(1) == y[::4]).mean())


def test_api_and_artifact_integrity(tmp_path):
    parts = {"fit": poems("fit"), "selection": poems("selection"), "calibration": []}
    train_fit(config("cnn"), parts, tmp_path / "run", model_factory=tiny_model)
    folder = tmp_path / "run/model"
    client = TestClient(create_app(folder, cpu_threads=1))
    assert client.get("/health").json()["status"] == "ok"
    assert "A padyamu, a prediction." in client.get("/").text
    assert 'id="theme"' in client.get("/").text
    payload = client.post("/api/predict", json={"text": "\n".join(parts["selection"][0].lines)}).json()
    assert payload["confidence_interval"] is None
    assert sum(c["probability"] for c in payload["classes"]) == pytest.approx(1.0)
    assert len(payload["lines"]) == 4
    assert client.post("/api/predict", json={"text": "one\ntwo"}).status_code == 422
    assert client.post("/api/predict", json={"text": "a" * 129}).status_code == 422
    (folder / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_package(folder)


def test_single_and_ensemble_are_calibrated_probability_averages(tmp_path):
    from chandassu.inference.collection import ModelCollection
    from chandassu.inference.predictor import Predictor

    parts = {"fit": poems("fit"), "selection": poems("selection"), "calibration": poems("calibration")}
    folders = []
    for seed in (17, 42):
        folder = tmp_path / f"seed-{seed}"
        train_fit(replace(config("cnn"), seed=seed), parts, folder, model_factory=tiny_model)
        folders.append(folder / "model")
    text = "\n".join(parts["selection"][0].lines)
    originals = [Predictor(folder).predict(text) for folder in folders]
    suite = ModelCollection(folders, strategy="ensemble")
    ensemble = suite.predict(text)
    expected = np.mean([[c["probability"] for c in result["classes"]] for result in originals], axis=0)
    assert [c["probability"] for c in ensemble["classes"]] == pytest.approx(expected)
    expected_lines = np.mean([[line["probabilities"] for line in result["lines"]] for result in originals], axis=0)
    assert np.array([line["probabilities"] for line in ensemble["lines"]]) == pytest.approx(expected_lines)
    assert ensemble["confidence_interval"] is None and ensemble["prediction_models"] == 2
    assert ensemble["lines"][0]["logits"] is None
    single = suite.predict(text, "seed-42")
    assert [c["probability"] for c in single["classes"]] == pytest.approx(
        [c["probability"] for c in originals[1]["classes"]]
    )
    assert single["seed"] == 42 and single["prediction_models"] == 1
    assert single["compared_seeds"] == 2
    client = TestClient(create_app(folders, strategy="ensemble"))
    assert client.get("/api/info").json()["default_strategy"] == "ensemble"
    assert client.post("/api/predict", json={"text": text, "strategy": "seed-17"}).json()["seed"] == 17
    assert client.post("/api/predict", json={"text": text, "strategy": "seed-73"}).status_code == 422
    with pytest.raises(ValueError, match="at least two"):
        ModelCollection(folders[:1], strategy="ensemble")
    with pytest.raises(ValueError, match="distinct seed"):
        ModelCollection([folders[0], folders[0]])
