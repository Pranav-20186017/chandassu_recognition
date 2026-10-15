"""Load the public eight-class poem CNN and classify a UTF-8 poem text file.

python scripts/load_hf_model.py --poem poem.txt --revision FULL_HF_COMMIT
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="sprx7767/chandassu-recognition")
    parser.add_argument("--revision", help="Immutable HF commit; if omitted, resolve main once and pin both loaders")
    parser.add_argument("--subfolder", help="Optional explicit package: cnn-eight-class")
    parser.add_argument("--poem", type=Path, required=True, help="Complete newline-separated poem in UTF-8")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args(argv)
    revision = args.revision or HfApi().model_info(args.repo_id, token=False).sha
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        parser.error("--revision must be a complete lowercase hexadecimal HF commit ID")
    options = dict(revision=revision, trust_remote_code=True, token=False)
    if args.subfolder:
        options["subfolder"] = args.subfolder
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, **options)
    model = AutoModelForSequenceClassification.from_pretrained(args.repo_id, **options).to(args.device).eval()
    if getattr(model.config, "prediction_unit", None) != "poem":
        parser.error("This script requires the eight-class whole-poem model; original four-class models classify lines")
    inputs = tokenizer(args.poem.read_text(encoding="utf-8"), padding=True, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        scores = model(**inputs).logits.double().softmax(-1)[0]
    print(f"Hugging Face revision: {revision}")
    print(f"Prediction: {model.config.id2label[int(scores.argmax())]} (uncalibrated score)")
    for i in scores.argsort(descending=True).tolist():
        print(f"{model.config.id2label[i]}: {float(scores[i]):.4%}")


if __name__ == "__main__":
    main()
