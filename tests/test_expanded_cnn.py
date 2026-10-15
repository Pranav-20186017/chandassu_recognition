import json

import numpy as np
import pytest
import torch

from chandassu.expanded import (
    LABELS,
    ExpandedPoem,
    OrderedPoemCNN,
    assert_parts,
    collate,
    encode,
    load_corpus,
    load_model,
)
from chandassu.models.encoding import CharacterVocabulary
from train_cnn_expanded import better, metrics, train


def example(label=LABELS[0], source="work", lines=None):
    return ExpandedPoem.from_dict(
        dict(
            poem_id=source + label,
            source=source,
            author="author",
            label=label,
            lines=lines or [f"అ ఆ {i}" for i in range(4)],
        )
    )


def test_ordered_cnn_masks_padding_and_accepts_long_body():
    torch.manual_seed(17)
    model = OrderedPoemCNN(30, width=8, kernel_sizes=(3, 5), dropout=0).eval()
    x = torch.randint(1, 30, (2, 4, 240))
    z = model(x)
    padded = nn_pad(x, 17)
    torch.testing.assert_close(z, model(padded), atol=1e-6, rtol=1e-5)
    assert z.shape == (2, 8)
    assert not torch.allclose(z, model(x.flip(1)))
    with pytest.raises(ValueError, match="four ordered"):
        model(x[:, :3])


def nn_pad(x, padding):
    return torch.nn.functional.pad(x, (0, padding))


def test_closure_is_provenance_and_capacity_never_truncates():
    row = dict(
        poem_id="sisa",
        source="sisa",
        author="author",
        label="సీసము",
        lines=["అ ఆ ఇ ఈ"] * 4,
        closing_geeti_lines=["ఇ ఈ ఉ ఊ"] * 4,
    )
    poem = ExpandedPoem.from_dict(row)
    vocabulary = CharacterVocabulary(30).fit(poem.lines)
    ids, y = collate(encode([poem], vocabulary))
    assert ids.shape == (1, 4, 7)
    assert y.item() == 7
    assert "ఉ" not in vocabulary.tokens
    with pytest.raises(ValueError, match="no truncation"):
        CharacterVocabulary(2).fit(poem.lines).encode(poem.lines[0])
    with pytest.raises(ValueError, match="four complete"):
        ExpandedPoem.from_dict(dict(row, lines=row["lines"] + row["closing_geeti_lines"]))


def test_closure_overlap_rejected_even_when_not_input():
    sisa = ExpandedPoem.from_dict(
        dict(
            poem_id="sisa",
            source="sisa",
            author="a",
            label="సీసము",
            lines=["అ ఆ ఇ ఈ"] * 4,
            closing_geeti_lines=["ఉ ఊ ఋ"] * 4,
        )
    )
    other = example(source="other", lines=["ఉ ఊ ఋ"] * 4)
    with pytest.raises(ValueError, match="closure"):
        assert_parts({"reserved": [sisa], "other": [other]})


def test_metrics_and_checkpoint_select_all_eight_classes():
    m = metrics(np.eye(8) * 10, list(range(8)))
    assert m["macro_f1"] == 1
    assert len(m["confusion_matrix"]) == 8
    best = dict(validation_poem_macro_f1=0.8, validation_ce=0.2)
    assert better(dict(validation_poem_macro_f1=0.9, validation_ce=0.5), best)
    assert not better(dict(validation_poem_macro_f1=0.7, validation_ce=0.1), best)
    assert better(dict(validation_poem_macro_f1=0.8, validation_ce=0.1), best)


def test_v3_frozen_roles_and_eight_label_coverage():
    from pathlib import Path

    if not Path("data/v3/manifest.json").exists():
        pytest.skip("Prepared v3 corpus not present")
    parts, manifest = load_corpus("data/v3")
    old = json.loads(Path("data/v2/splits/train_expansion.json").read_text())["ids"]
    for role, ids in old.items():
        assert set(ids) <= {p.poem_id for p in parts[role]}
    assert {p.poem_id for p in parts["test"] if p.metadata["cohort"] == "v2_existing"} == {
        json.loads(line)["poem_id"] for line in Path("data/v2/test.jsonl").open()
    }
    assert manifest["labels"] == list(LABELS)
    import re

    for rows in parts.values():
        for poem in rows:
            if poem.metadata["cohort"] == "new_source_annotation":
                assert not any(re.search("[A-Za-z<>]", text) for text in poem.lines)


def test_small_training_package_and_completed_resume(tmp_path):
    from chandassu.io import atomic_json, sha256

    folder = tmp_path / "data"
    folder.mkdir()
    files, count_map = {}, {}
    from chandassu.expanded import counts

    for role in ["fit", "selection", "calibration", "test"]:
        rows = []
        for i, label in enumerate(LABELS):
            # Telugu codepoints differ across roles and labels, avoiding layout duplicates.
            offset = ["fit", "selection", "calibration", "test"].index(role) * 10
            text = chr(0x0C15 + offset) + chr(0x0C15 + i)
            rows.append(
                dict(
                    poem_id=f"{role}-{i}",
                    source=role,
                    author="a",
                    label=label,
                    lines=[text] * 4,
                    cohort="new_source_annotation",
                )
            )
        path = folder / f"{role}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[path.name] = sha256(path)
        count_map[role] = counts([ExpandedPoem.from_dict(r) for r in rows])
    atomic_json(folder / "manifest.json", dict(schema_version=3, labels=LABELS, files=files, counts=count_map))
    config = json.loads(__import__("pathlib").Path("configs/cnn_eight_class.json").read_text())
    config.update(
        data_dir=str(folder),
        output_dir=str(tmp_path / "run"),
        epochs=1,
        minimum_epochs=1,
        width=4,
        batch_poems=8,
        max_tokens=16,
        device="cpu",
        precision="fp32",
        cpu_threads=1,
    )
    first = train(config)
    model, vocabulary, saved = load_model(tmp_path / "run/model")
    assert saved["labels"] == list(LABELS)
    assert model.classifier[-1].out_features == 8
    assert (tmp_path / "run/test.json").is_file()
    assert train(config, resume=True)["run_id"] == first["run_id"]
    with pytest.raises(ValueError, match="identical"):
        train(dict(config, learning_rate=0.002), resume=True)
