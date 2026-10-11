"""Export existing safe packages for the Transformers Auto interfaces."""

import json
import shutil
from pathlib import Path

from . import LABELS
from .io import atomic_json, sha256


def export_hf(source, destination):
    """Preserve weights and native loading; add self-contained Hub classes."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or (destination.exists() and any(destination.iterdir())):
        raise ValueError("Export destination must be separate and empty")
    manifest = json.loads((source / "manifest.json").read_text())
    required = {"config.json", "vocabulary.json", "metadata.json", "model.safetensors"}
    if manifest.get("schema_version") != 1 or not required <= manifest["files"].keys():
        raise ValueError("Incomplete or unsupported source package")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(source / name) != expected:
            raise ValueError(f"Model package checksum mismatch: {name}")
    config = json.loads((source / "config.json").read_text())
    vocabulary = json.loads((source / "vocabulary.json").read_text())
    family = config["model_type"]
    if family not in {"cnn", "byt5"} or config["labels"] != list(LABELS):
        raise ValueError("Unsupported model family or labels")
    if config["max_tokens"] != vocabulary["max_tokens"]:
        raise ValueError("Encoding limit mismatch")
    stem = "ChandassuCNN" if family == "cnn" else "ChandassuByT5"
    model_class = f"hf_model.{stem}ForSequenceClassification"
    config.update(
        architectures=[f"{stem}ForSequenceClassification"],
        calibration_temperature=config["temperature"],
        auto_map={
            "AutoConfig": f"hf_model.{stem}Config",
            "AutoModel": model_class,
            "AutoModelForSequenceClassification": model_class,
        },
        id2label={str(i): label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
        pad_token_id=0,
    )
    if family == "cnn":
        config["vocab_size"] = len(vocabulary["tokens"]) + 2
    else:
        if "encoder_config.json" not in manifest["files"]:
            raise ValueError("Missing encoder configuration")
        config["encoder_config"] = json.loads((source / "encoder_config.json").read_text())
        config["vocab_size"] = config["encoder_config"]["vocab_size"]
    shutil.copytree(source, destination, dirs_exist_ok=True)
    atomic_json(destination / "config.json", config)
    atomic_json(
        destination / "tokenizer_config.json",
        {
            "tokenizer_class": "ChandassuTokenizer",
            "auto_map": {"AutoTokenizer": ["hf_model.ChandassuTokenizer", None]},
            "model_max_length": config["max_tokens"],
            "padding_side": "right",
        },
    )
    shutil.copyfile(Path(__file__).with_name("hf_model.py"), destination / "hf_model.py")
    names = set(manifest["files"]) | {"tokenizer_config.json", "hf_model.py"}
    atomic_json(
        destination / "manifest.json",
        {"schema_version": 1, "files": {name: sha256(destination / name) for name in sorted(names)}},
    )
    (destination / "README.md").write_text(
        f"""---
language: te
license: other
license_name: chandassu-citation
license_link: LICENSE
library_name: transformers
pipeline_tag: text-classification
tags: [telugu, poetry, metre, {family}]
---
# Chandassu {family.upper()}

Use `AutoTokenizer` and `AutoModelForSequenceClassification` with
`subfolder=\"{family}\"` and `trust_remote_code=True`.
The model returns raw logits for each line. Divide logits by `model.config.calibration_temperature`
before applying softmax. For a four-line poem, average the four probability vectors.
See the [main model card](https://huggingface.co/sprx7767/chandassu-recognition)
for code, results, limits, and citation terms.
The project loader and `inference.py` also support this package.
"""
    )
    return destination
