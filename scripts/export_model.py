"""Convert a frozen H100 checkpoint to a self-contained, safe inference package."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import T5Config

from chandassu.artifacts import save_package
from chandassu.config import resolve
from chandassu.data import Corpus, fingerprint
from chandassu.io import sha256
from chandassu.models.encoding import vocabulary_for
from chandassu.models.factory import build_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model_type", choices=["cnn", "byt5"], required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data_dir", type=Path, default=Path("data/v1"))
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Export destination must be empty; existing packages are preserved")
    report = json.loads((args.bundle / "final_report.json").read_text())
    identity = json.loads((args.bundle / "identity.json").read_text())
    recipe = json.loads(
        (args.bundle / f"final_models/{args.model_type}/seed-{args.seed}/final_recipe.json").read_text()
    )
    record = report["models"][f"{args.model_type}/seed-{args.seed}"]
    checkpoint = args.bundle / Path(record["checkpoint"]).relative_to(report["outputs"])
    if sha256(checkpoint) != recipe["checkpoint_sha256"]:
        raise ValueError("Original checkpoint hash does not match frozen recipe")
    corpus = Corpus(args.data_dir)
    parts = corpus.final_parts("splits/final_h100.json", 0.15, 142)
    fit_key = {
        "run_sha256": recipe["run_sha256"],
        "arm": args.model_type,
        "seed": args.seed,
        "fit_sha256": fingerprint(parts["fit"]),
        "selection_sha256": fingerprint(parts["selection"]),
    }
    original_fit_id = hashlib.sha256(
        (json.dumps(fit_key, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    ).hexdigest()
    if original_fit_id != recipe["fit_sha256"]:
        raise ValueError("Prepared fitting texts do not match checkpoint fingerprint")
    config = resolve([f"--model_type={args.model_type}", f"--seed={args.seed}"])
    search = json.loads((args.bundle / f"final_models/{args.model_type}/search/search.json").read_text())
    rate = next(r["rate"] for r in search["candidates"] if r["candidate"] == search["winner"])
    settings = dict(asdict(config), learning_rate=rate)
    vocabulary = vocabulary_for(args.model_type, config.max_tokens, parts["fit"])
    original_vocabulary = json.loads((checkpoint.parent / "vocabulary.json").read_text())
    if original_vocabulary["tokens"] != getattr(vocabulary, "tokens", None):
        raise ValueError("Recovered vocabulary differs from the saved H100 vocabulary")
    encoder_config = (
        T5Config.from_pretrained(config.base_model, revision=config.model_revision).to_dict()
        if args.model_type == "byt5"
        else None
    )
    torch.set_num_threads(4)
    model = build_model(settings, vocabulary, encoder_config=encoder_config, pretrained=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["fit_sha256"] != recipe["fit_sha256"] or saved["epoch"] != record["selected_epoch"]:
        raise ValueError("Checkpoint metadata mismatch")
    model.load_state_dict(saved["model"], strict=True)
    development = json.loads((args.bundle / "development_metrics.json").read_text())
    development_means = {
        level: sum(
            development[f"{args.model_type}/seed-{seed}"]["label_cohorts"]["source_annotation"][level]["aggregates"][
                "macro"
            ]["f1"]
            for seed in (17, 42, 73)
        )
        / 3
        for level in ("poem", "line")
    }

    def performance(metrics, rejection):
        return {
            "development_poem_macro_f1_three_seed_mean": development_means["poem"],
            "development_line_macro_f1_three_seed_mean": development_means["line"],
            "historical_poems": metrics["poem"]["observations"],
            "historical_poem_macro_f1": metrics["poem"]["aggregates"]["macro"]["f1"],
            "historical_line_macro_f1": metrics["line"]["aggregates"]["macro"]["f1"],
            "historical_other_metre_false_acceptance": rejection["other_false_acceptance"],
        }

    ensemble = report["ensembles"][f"{args.model_type}_ensemble"]
    ensemble_hashes = {
        str(seed): json.loads(
            (args.bundle / f"final_models/{args.model_type}/seed-{seed}/final_recipe.json").read_text()
        )["checkpoint_sha256"]
        for seed in (17, 42, 73)
    }
    metadata = {
        "seed": args.seed,
        "selected_epoch": saved["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "original_recipe": recipe,
        "original_identity": identity,
        "source_checkpoint_sha256": sha256(checkpoint),
        "partitions": {
            role: {"sha256": fingerprint(poems), "poem_ids": [p.poem_id for p in poems]}
            for role, poems in parts.items()
        },
        "historical_metrics": record["historical_metrics"],
        "performance": performance(record["historical_metrics"], record["historical_rejection"]),
        "published_ensemble": {
            "checkpoint_hashes": ensemble_hashes,
            "performance": performance(ensemble["historical_metrics"], ensemble["historical_rejection"]),
        },
        "historical_test_used_for_selection": False,
        "rejection_threshold_feasible": False,
        "confidence_interval": None,
        "selection_policy": "Seed chosen in advance; family rate selected on validation. Historical test diagnostic only.",
    }
    save_package(args.output_dir, model, settings, vocabulary, recipe["temperature"], metadata)
    (args.output_dir / "LICENSE").write_text(Path("LICENSE").read_text())
    (args.output_dir / "NOTICE").write_text(Path("NOTICE").read_text())
    card = f"""---
license: apache-2.0
language: te
tags: [telugu, poetry, metre, {args.model_type}]
library_name: pytorch
pipeline_tag: text-classification
---
# Chandassu {args.model_type.upper()}

Four-class Telugu poetic metre classifier, trained on NVIDIA H100 NVL.
Seed {args.seed}; selected epoch {saved["epoch"]}; {metadata["parameters"]:,} parameters.
Calibration temperature: {recipe["temperature"]}. Mean calibrated line probabilities classify a four-line poem.

Classes: ఉత్పలమాల, చంపకమాల, మత్తేభము, శార్దూలం (in this order).

Load this package with `chandassu.artifacts.load_package`, or run the repository's
`inference.py --model_dir PATH`. This custom encoder/head package does not implement
the generic Transformers AutoModel interface. No pickle is required for inference.

Training: 6,517 poems; separate calibration: 908; selection: 1,859.
Historical diagnostic: 2,234 poems, already inspected during earlier project development.
See metadata.json and the repository results/v1 for full metrics and provenance.
This is a restricted classifier: it cannot reliably reject other metres or provide
prosodic explanations. Scores are calibrated probabilities, not confidence intervals.
Source edition/transcription rights are not independently cleared; dataset terms
are separate from the Apache-2.0 code/weight license. See the repository data/LICENSE.md.
"""
    if args.model_type == "byt5":
        card += f"\nBase encoder: google/byt5-small (Apache-2.0), revision {config.model_revision}.\n"
    (args.output_dir / "README.md").write_text(card)
    print(f"Exported {args.model_type}, seed {args.seed}: {args.output_dir}")


if __name__ == "__main__":
    main()
