import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from chandassu.artifacts import save_package
from chandassu.config import TrainConfig
from chandassu.expanded import LABELS, OrderedPoemCNN
from chandassu.inference.expanded import ExpandedPredictor
from chandassu.inference.server import create_app
from chandassu.io import atomic_json, sha256
from chandassu.models.direct import TeluguMetreCNN
from chandassu.models.encoding import CharacterVocabulary


@pytest.fixture
def packages(tmp_path):
    halves = ["అ ఆ ఇ", "ఈ ఉ ఊ", "ఎ ఏ ఐ", "ఒ ఓ ఔ", "క ఖ గ", "ఘ చ ఛ", "జ ఝ ట", "ఠ డ ఢ"]
    padas = [" ".join(halves[i : i + 2]) for i in range(0, 8, 2)]
    vocabulary = CharacterVocabulary(512).fit(padas)
    folder = tmp_path / "expanded/model"
    folder.mkdir(parents=True)
    torch.manual_seed(17)
    model = OrderedPoemCNN(len(vocabulary.tokens) + 2, width=8)
    save_file(model.state_dict(), str(folder / "model.safetensors"))
    atomic_json(
        folder / "config.json",
        dict(
            architecture="ordered_poem_cnn_v1",
            labels=LABELS,
            max_tokens=512,
            width=8,
            kernel_sizes=[3, 5, 7],
            dropout=0.2,
            run_id="test-run",
            calibrated=False,
        ),
    )
    atomic_json(folder / "vocabulary.json", vocabulary.tokens)
    atomic_json(
        folder / "manifest.json",
        dict(files={name: sha256(folder / name) for name in ["model.safetensors", "config.json", "vocabulary.json"]}),
    )
    atomic_json(folder.parent / "run.json", dict(run_id="test-run", contract=dict(config=dict(seed=17))))
    originals = []
    for seed in [17, 42]:
        legacy_folder = tmp_path / f"original-{seed}"
        config = TrainConfig(width=8, seed=seed, max_tokens=512)
        save_package(
            legacy_folder,
            TeluguMetreCNN(len(vocabulary.tokens) + 2, width=8),
            asdict(config),
            vocabulary,
            1.0,
            dict(seed=seed),
        )
        originals.append(legacy_folder)
    return folder, originals, halves, padas


def test_new_api_matches_direct_joint_prediction_and_excludes_closure(packages):
    folder, originals, halves, padas = packages
    client = TestClient(create_app([folder, *originals], cpu_threads=1))
    info = client.get("/api/info").json()
    assert info["labels"] == list(LABELS)
    assert info["default_strategy"] == "eight-class-seed-17"
    assert {s["id"] for s in info["strategies"]} == {"eight-class-seed-17", "seed-17", "seed-42", "ensemble"}
    direct = ExpandedPredictor(folder, cpu_threads=1).predict("\n".join(padas))
    full = client.post("/api/predict", json={"text": "\n".join(padas)}).json()
    assert len(full["classes"]) == 8 and full["prediction_models"] == 1
    assert full["calibrated"] is False and full["line_agreement"] is None
    assert all(line["probabilities"] is None and line["logits"] is None for line in full["lines"])
    np.testing.assert_allclose(
        [c["probability"] for c in full["classes"]], [c["probability"] for c in direct["classes"]]
    )
    closure = ["unknown" * 100] * 4
    for text in ["\n".join(halves), "\n".join(halves + closure)]:
        response = client.post("/api/predict", json={"text": text})
        assert response.status_code == 200
        assert [c["probability"] for c in response.json()["classes"]] == [c["probability"] for c in full["classes"]]
    assert response.json()["closing_geeti_lines_excluded"] == closure
    assert response.json()["line_count"] == 4 and response.json()["input_line_count"] == 12


def test_original_single_line_and_ensemble_remain_available(packages):
    folder, originals, _, padas = packages
    client = TestClient(create_app([folder, *originals], cpu_threads=1))
    old = TestClient(create_app(originals, cpu_threads=1))
    for text, strategy in [(padas[0], "seed-17"), ("\n".join(padas), "ensemble")]:
        new_response = client.post("/api/predict", json=dict(text=text, strategy=strategy))
        old_response = old.post("/api/predict", json=dict(text=text, strategy=strategy))
        assert new_response.status_code == old_response.status_code == 200
        assert new_response.json()["classes"] == old_response.json()["classes"]
        assert new_response.json()["lines"] == old_response.json()["lines"]
        assert len(new_response.json()["classes"]) == 4
    assert client.post("/api/predict", json=dict(text=padas[0])).status_code == 422
    assert client.post("/api/predict", json=dict(text="\n".join(padas + ["extra"]))).status_code == 422
    assert client.post("/api/predict", json=dict(text="\n".join(["అ" * 513] * 4))).status_code == 422
    assert client.post("/api/predict", json=dict(text="\n".join(padas), strategy="not-loaded")).status_code == 422


def test_expanded_package_integrity_and_compatible_ensemble_selection(packages):
    folder, originals, _, _ = packages
    client = TestClient(create_app([folder, *originals], strategy="ensemble", cpu_threads=1))
    assert client.get("/api/info").json()["default_strategy"] == "ensemble"
    with pytest.raises(ValueError, match="two compatible"):
        create_app(folder, strategy="ensemble", cpu_threads=1)
    config = json.loads((folder / "config.json").read_text())
    config["width"] = 9
    atomic_json(folder / "config.json", config)
    with pytest.raises(ValueError, match="checksum"):
        create_app(folder, cpu_threads=1)
