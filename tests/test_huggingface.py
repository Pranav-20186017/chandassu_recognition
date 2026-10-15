"""The public Auto interfaces must preserve the native package's encoding and scores."""

import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, T5Config

from chandassu.artifacts import load_package, save_package
from chandassu.config import TrainConfig
from chandassu.hub import export_hf
from chandassu.models.encoding import vocabulary_for
from chandassu.models.factory import build_model


@pytest.fixture(autouse=True)
def module_cache(tmp_path, monkeypatch):
    import transformers.dynamic_module_utils as dynamic

    monkeypatch.setattr(dynamic, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    torch.set_num_threads(1)


def package(tmp_path, arm):
    settings = asdict(TrainConfig(model_type=arm, width=8, max_tokens=128))
    vocabulary = vocabulary_for(arm, 128, saved={"అ": 2, "ఆ": 3, " ": 4} if arm == "cnn" else None)
    encoder = T5Config(vocab_size=384, d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8)
    original = build_model(settings, vocabulary, encoder_config=encoder.to_dict(), pretrained=False).eval()
    native = save_package(tmp_path / "native", original, settings, vocabulary, 1.37, {})
    exported = export_hf(native, tmp_path / arm)
    return original, vocabulary, exported


@pytest.mark.parametrize("arm", ["cnn", "byt5"])
def test_auto_encoding_logits_calibration_and_roundtrip(tmp_path, arm):
    original, vocabulary, exported = package(tmp_path, arm)
    tokenizer = AutoTokenizer.from_pretrained(exported, trust_remote_code=True, local_files_only=True)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        exported, trust_remote_code=True, local_files_only=True, output_loading_info=True
    )
    assert not any(info.values()), info
    lines = ["అఆ", " \ufeffఆ  అ \t", "అ్\u200c \u200d", "[PAD] [UNK] </s> e\u0301"]
    encoded = [vocabulary.encode(line) for line in lines]
    assert [tokenizer.encode(line) for line in lines] == encoded
    inputs = tokenizer(lines, padding=True, return_tensors="pt")
    with torch.inference_mode():
        expected = original(inputs["input_ids"])
        actual = model(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.config.id2label == {i: label for i, label in enumerate(model.config.labels)}
    torch.testing.assert_close(
        (actual.double() / model.config.calibration_temperature).softmax(-1).mean(0),
        (expected.double() / 1.37).softmax(-1).mean(0),
        rtol=0,
        atol=0,
    )
    # A shorter line must retain its score when padded into a mixed-length batch.
    with torch.inference_mode():
        single = model(**tokenizer(lines[0], return_tensors="pt")).logits[0]
    torch.testing.assert_close(single, actual[0], rtol=1e-5, atol=1e-6)
    native_model, _, _, _ = load_package(exported)
    with torch.inference_mode():
        torch.testing.assert_close(native_model(inputs["input_ids"]), expected, rtol=0, atol=0)
    loss = model(**inputs, labels=torch.tensor([0, 1, 2, 3])).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.classifier.weight.grad is not None
    with pytest.raises(ValueError, match="[Tt]runcation"):
        tokenizer(lines, truncation=True, max_length=2)
    with pytest.raises(ValueError, match="[Ll]ine"):
        tokenizer("అ" * 129)
    with pytest.raises(ValueError, match="separately"):
        tokenizer("అ", text_pair="ఆ")
    saved = tmp_path / "roundtrip"
    model.save_pretrained(saved, safe_serialization=True)
    tokenizer.save_pretrained(saved)
    reloaded, info = AutoModel.from_pretrained(
        saved, trust_remote_code=True, local_files_only=True, output_loading_info=True
    )
    assert not any(info.values()), info
    assert reloaded.config.calibration_temperature == 1.37
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(**inputs).logits, expected, rtol=0, atol=0)
    assert AutoTokenizer.from_pretrained(saved, trust_remote_code=True).encode(lines[-1]) == encoded[-1]


@pytest.mark.parametrize("arm", ["cnn", "byt5"])
def test_standalone_subfolder_loading_without_project_imports(tmp_path, arm):
    _, vocabulary, exported = package(tmp_path, arm)
    (tmp_path / "hf_model.py").write_bytes((exported / "hf_model.py").read_bytes())
    script = r"""
import importlib.abc, json, sys
class BlockProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "chandassu" or fullname.startswith("chandassu."):
            raise ImportError("The standalone loader must not import the project")
        if sys.argv[2] == "cnn" and fullname == "transformers.models.t5.modeling_t5":
            raise ImportError("T5 is unavailable; loading the CNN must not require it")
sys.meta_path.insert(0, BlockProject())
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
torch.set_num_threads(1)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], subfolder=sys.argv[2], trust_remote_code=True, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1], subfolder=sys.argv[2], trust_remote_code=True, local_files_only=True)
inputs = tokenizer("అఆ", return_tensors="pt")
with torch.inference_mode():
    assert model(**inputs).logits.shape == (1, 4)
assert not any(k == "chandassu" or k.startswith("chandassu.") for k in sys.modules)
if sys.argv[2] == "cnn":
    assert "transformers.models.t5.modeling_t5" not in sys.modules
print(json.dumps(tokenizer.encode("అఆ")))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path), arm],
        cwd=tmp_path,
        env=dict(os.environ, HF_MODULES_CACHE=str(tmp_path / "isolated-modules"), HF_HUB_OFFLINE="1"),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == vocabulary.encode("అఆ")
