"""Export eight-class native weights with a standalone Transformers poem loader."""

import json
import shutil
from pathlib import Path

from .expanded import LABELS, load_model
from .io import atomic_json, sha256


def export_expanded_hf(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or (destination.exists() and any(destination.iterdir())):
        raise ValueError("Export destination must be separate and empty")
    load_model(source)  # Verify the native manifest and complete state dictionary.
    manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "config.json").read_text())
    vocabulary = json.loads((source / "vocabulary.json").read_text())
    config.update(
        model_type="chandassu_poem_cnn",
        vocab_size=len(vocabulary) + 2,
        architectures=["ChandassuPoemCNNForSequenceClassification"],
        auto_map={
            "AutoConfig": "hf_poem.ChandassuPoemCNNConfig",
            "AutoModel": "hf_poem.ChandassuPoemCNNForSequenceClassification",
            "AutoModelForSequenceClassification": "hf_poem.ChandassuPoemCNNForSequenceClassification",
        },
        id2label={str(i): label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
        pad_token_id=0,
        prediction_unit="poem",
    )
    names = set(manifest["files"]) | {"hf_poem.py", "tokenizer_config.json", "metadata.json"}
    metadata = {}
    for name in ("run.json", "result.json", "test.json"):
        path = source.parent / name
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("run_id") != config["run_id"]:
                raise ValueError("Export report run identity mismatch")
            metadata[name] = report
    config["seed"] = metadata.get("run.json", {}).get("contract", {}).get("config", {}).get("seed")
    shutil.copytree(source, destination)
    atomic_json(destination / "config.json", config)
    atomic_json(destination / "metadata.json", metadata)
    atomic_json(
        destination / "tokenizer_config.json",
        dict(
            tokenizer_class="ChandassuPoemTokenizer",
            auto_map={"AutoTokenizer": ["hf_poem.ChandassuPoemTokenizer", None]},
            model_max_length=config["max_tokens"],
            padding_side="right",
        ),
    )
    shutil.copyfile(Path(__file__).with_name("hf_poem.py"), destination / "hf_poem.py")
    atomic_json(destination / "manifest.json", dict(files={name: sha256(destination / name) for name in sorted(names)}))
    (destination / "README.md").write_text("""---
language: te
license: other
license_name: chandassu-citation
license_link: LICENSE
library_name: transformers
pipeline_tag: text-classification
tags: [telugu, poetry, metre, cnn, eight-class]
---
# Eight-class ordered poem CNN

Fresh seed-17 CNN, selected at epoch 11. Test poem Macro F1 **94.84%**;
accuracy **97.90%**. Ataveladi F1 **69.34%** is the principal weakness.

Use `AutoTokenizer` and `AutoModelForSequenceClassification` with
`subfolder="cnn-eight-class"` and `trust_remote_code=True`. Pass a complete
newline-separated poem to the tokenizer, not individual lines. Tensors have shape
`[batch, 4 ordered padas, tokens]`; output is one eight-class logit vector per poem.
Scores are **uncalibrated**. Do not average these logits as line predictions.

Four full padas, eight Seesamu body half-lines, or eight body half-lines plus four
closing geeti lines are supported. The closing geeti never enters the CNN.
Maximum length is 512 characters per full pada; input is never silently truncated.
A batch is a list of complete poem strings. Standard line-level pipelines are
unsupported; use the Auto interfaces and explicit whole-poem inputs.

The native project loader and `inference.py` also support this exact package.
See the [main model card](https://huggingface.co/sprx7767/chandassu-recognition)
for usage, eight-label mapping, evaluation, source limitations and citation terms.
""")
    return destination
