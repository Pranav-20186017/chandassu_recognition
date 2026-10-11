"""Self-contained model packages shared by training, CLI, and the server."""

import json
import os
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file

from . import LABELS
from .io import atomic_json, sha256
from .models.encoding import vocabulary_for
from .models.factory import build_model


def save_package(folder, model, config, vocabulary, temperature, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not 0 < temperature < float("inf"):
        raise ValueError("Temperature must be finite and positive")
    config = dict(config, labels=list(LABELS), temperature=temperature, artifact_schema=1)
    atomic_json(folder / "config.json", config)
    atomic_json(
        folder / "vocabulary.json",
        {
            "kind": "character" if config["model_type"] == "cnn" else "utf8_bytes",
            "max_tokens": vocabulary.max_tokens,
            "tokens": getattr(vocabulary, "tokens", None),
        },
    )
    names = ["config.json", "vocabulary.json", "metadata.json", "model.safetensors"]
    if config["model_type"] == "byt5":
        atomic_json(folder / "encoder_config.json", model.encoder.config.to_dict())
        names.append("encoder_config.json")
    atomic_json(folder / "metadata.json", metadata)
    fd, temporary = tempfile.mkstemp(prefix=".weights.", dir=folder)
    os.close(fd)
    try:
        # Clone aliases such as T5 shared/embed_tokens so every state key is retained.
        state = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
        save_file(state, temporary)
        os.replace(temporary, folder / "model.safetensors")
    finally:
        Path(temporary).unlink(missing_ok=True)
    atomic_json(
        folder / "manifest.json", {"schema_version": 1, "files": {name: sha256(folder / name) for name in names}}
    )
    return folder


def load_package(folder, device="cpu"):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported model package")
    required = {"config.json", "vocabulary.json", "metadata.json", "model.safetensors"}
    if not required <= manifest["files"].keys():
        raise ValueError("Incomplete model package manifest")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(folder / name) != expected:
            raise ValueError(f"Model package checksum mismatch: {name}")
    config = json.loads((folder / "config.json").read_text())
    if config["artifact_schema"] != 1 or config["labels"] != list(LABELS):
        raise ValueError("Unsupported artifact schema or label order")
    if not 0 < config["temperature"] < float("inf"):
        raise ValueError("Invalid model temperature")
    saved = json.loads((folder / "vocabulary.json").read_text())
    if config["model_type"] not in {"cnn", "byt5"} or config["max_tokens"] != saved["max_tokens"]:
        raise ValueError("Invalid model type or encoding limit")
    if config["model_type"] == "byt5" and "encoder_config.json" not in manifest["files"]:
        raise ValueError("ByT5 encoder configuration missing from manifest")
    vocabulary = vocabulary_for(config["model_type"], saved["max_tokens"], saved=saved["tokens"])
    encoder_config = (
        json.loads((folder / "encoder_config.json").read_text()) if config["model_type"] == "byt5" else None
    )
    model = build_model(config, vocabulary, encoder_config=encoder_config, pretrained=False)
    model.load_state_dict(load_file(folder / "model.safetensors"))
    model.to(device).eval()
    return model, vocabulary, config, json.loads((folder / "metadata.json").read_text())
