"""Eight-class publication must preserve whole-poem encoding and exact native logits."""

import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from chandassu.expanded import LABELS, OrderedPoemCNN, collate, load_model
from chandassu.hub_expanded import export_expanded_hf
from chandassu.inference.expanded import ExpandedPredictor
from chandassu.io import atomic_json, sha256
from chandassu.models.encoding import CharacterVocabulary


@pytest.fixture
def exported(tmp_path, monkeypatch):
    import transformers.dynamic_module_utils as dynamic

    monkeypatch.setattr(dynamic, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    torch.set_num_threads(1)
    halves = ["అ ఆ", "ఇ ఈ", "ఉ ఊ", "ఋ ఎ", "ఏ ఐ", "ఒ ఓ", "ఔ క", "ఖ గ"]
    padas = [" ".join(halves[i : i + 2]) for i in range(0, 8, 2)]
    vocab = CharacterVocabulary(64).fit(padas)
    folder = tmp_path / "native/model"
    folder.mkdir(parents=True)
    torch.manual_seed(17)
    model = OrderedPoemCNN(len(vocab.tokens) + 2, width=8)
    save_file(model.state_dict(), str(folder / "model.safetensors"))
    atomic_json(
        folder / "config.json",
        dict(
            architecture="ordered_poem_cnn_v1",
            labels=LABELS,
            max_tokens=64,
            width=8,
            kernel_sizes=[3, 5, 7],
            dropout=0.2,
            run_id="hf-test",
            calibrated=False,
        ),
    )
    atomic_json(folder / "vocabulary.json", vocab.tokens)
    atomic_json(
        folder / "manifest.json",
        dict(files={name: sha256(folder / name) for name in ["model.safetensors", "config.json", "vocabulary.json"]}),
    )
    atomic_json(folder.parent / "run.json", dict(run_id="hf-test", contract=dict(config=dict(seed=17))))
    out = export_expanded_hf(folder, tmp_path / "hub/cnn-eight-class")
    return folder, out, padas, halves


def test_auto_poem_encoding_logits_and_roundtrip(exported, tmp_path):
    native, out, padas, halves = exported
    tokenizer = AutoTokenizer.from_pretrained(out, trust_remote_code=True, local_files_only=True)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        out, trust_remote_code=True, local_files_only=True, output_loading_info=True
    )
    assert not any(info.values()), info
    native_model, vocab, _ = load_model(native)
    assert sha256(out / "model.safetensors") == sha256(native / "model.safetensors")
    poems = ["\n".join(padas), "\n".join([padas[0] + " అ", *padas[1:]])]
    inputs = tokenizer(poems, padding=True, return_tensors="pt")
    expected_ids, _ = collate([([vocab.encode(line) for line in poem.splitlines()], 0) for poem in poems])
    torch.testing.assert_close(inputs["input_ids"], expected_ids, rtol=0, atol=0)
    assert inputs["input_ids"].shape[:2] == (2, 4)
    with torch.inference_mode():
        expected = native_model(expected_ids)
        actual = model(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 8) and model.config.calibrated is False
    for text in ["\n".join(halves), "\n".join(halves + ["q" * 200] * 4)]:
        with torch.inference_mode():
            logits = model(**tokenizer(text)).logits
        torch.testing.assert_close(logits[0], actual[0], rtol=1e-5, atol=1e-6)
    assert tokenizer.encode(poems[0]) == [vocab.encode(line) for line in padas]
    for kwargs in [dict(truncation=True), dict(max_length=2), dict(text_pair="అ")]:
        with pytest.raises(ValueError):
            tokenizer(poems[0], **kwargs)
    with pytest.raises(ValueError, match="four full"):
        tokenizer(padas[0])
    with pytest.raises(ValueError, match="no truncation"):
        tokenizer("\n".join(["అ" * 65] * 4))
    with pytest.raises(ValueError, match="Attention mask"):
        model(input_ids=inputs["input_ids"], attention_mask=torch.zeros_like(inputs["attention_mask"]))
    assert torch.isfinite(model(**inputs, labels=torch.tensor([0, 7])).loss)
    saved = tmp_path / "roundtrip"
    model.save_pretrained(saved, safe_serialization=True)
    tokenizer.save_pretrained(saved)
    reloaded = AutoModel.from_pretrained(saved, trust_remote_code=True, local_files_only=True)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(**inputs).logits, actual, rtol=0, atol=0)
    token_again = AutoTokenizer.from_pretrained(saved, trust_remote_code=True, local_files_only=True)
    torch.testing.assert_close(token_again(poems)["input_ids"], inputs["input_ids"])
    # The native server reads exported metadata even away from its original run folder.
    adapter = ExpandedPredictor(out, cpu_threads=1)
    assert adapter.info()["seed"] == 17
    assert adapter.predict(poems[0])["raw_poem_logits"] == pytest.approx(actual[0].tolist(), abs=1e-6)


def test_standalone_loading_without_project_or_t5(exported, tmp_path):
    _, out, padas, _ = exported
    root = out.parent
    (root / "hf_poem.py").write_bytes((out / "hf_poem.py").read_bytes())
    script = r"""
import importlib.abc,json,sys
class BlockProject(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='chandassu' or fullname.startswith('chandassu.') or fullname=='transformers.models.t5.modeling_t5':
            raise ImportError('Standalone CNN cannot import project or T5')
sys.meta_path.insert(0,BlockProject())
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
torch.set_num_threads(1)
kw=dict(subfolder='cnn-eight-class',trust_remote_code=True,local_files_only=True)
tokenizer=AutoTokenizer.from_pretrained(sys.argv[1],**kw)
model=AutoModelForSequenceClassification.from_pretrained(sys.argv[1],**kw)
with torch.inference_mode():
    assert model(**tokenizer(sys.argv[2])).logits.shape==(1,8)
assert not any(name=='chandassu' or name.startswith('chandassu.') for name in sys.modules)
print('standalone eight-class loading verified')
"""
    r = subprocess.run(
        [sys.executable, "-I", "-c", script, str(root), "\n".join(padas)],
        cwd=tmp_path,
        env=dict(os.environ, HF_MODULES_CACHE=str(tmp_path / "isolated"), HF_HUB_OFFLINE="1"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "verified" in r.stdout


def test_hub_cli_downloads_only_the_selected_package(exported, monkeypatch):
    import inference

    _, out, _, _ = exported
    requests = []

    def download(repo, filename, **kwargs):
        assert filename == "manifest.json" and kwargs["subfolder"] == "cnn-eight-class"
        return str(out / "manifest.json")

    def snapshot(repo, **kwargs):
        requests.extend(kwargs["allow_patterns"])
        return str(out.parent)

    started = []
    monkeypatch.setattr(inference, "hf_hub_download", download)
    monkeypatch.setattr(inference, "snapshot_download", snapshot)
    monkeypatch.setattr(inference.uvicorn, "run", lambda app, **kw: started.append(app))
    inference.main(["--model_id", "example/model", "--revision", "a" * 40, "--subfolder", "cnn-eight-class"])
    assert started[0].state.predictor.info()["labels"] == list(LABELS)
    assert "cnn-eight-class/hf_poem.py" in requests
    assert all(name.startswith("cnn-eight-class/") and "*" not in name for name in requests)
