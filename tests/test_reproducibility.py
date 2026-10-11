import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import T5Config

from chandassu import LABELS
from chandassu.artifacts import load_package
from chandassu.config import TrainConfig, resolve
from chandassu.data import Corpus, Poem, assert_isolated, normalize_text, plain_folds, work_groups
from chandassu.evaluation.metrics import fit_temperature, line_poem_metrics, softmax
from chandassu.inference.server import create_app
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
