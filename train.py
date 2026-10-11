"""Train CNN or ByT5 with flat, validated arguments."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from chandassu.config import resolve
from chandassu.data import Corpus, plain_folds
from chandassu.io import atomic_json
from chandassu.training.engine import train_fit


def main(argv=None):
    config = resolve(argv)
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2), flush=True)
    corpus = Corpus(config.data_dir)
    output = Path(config.output_dir)
    if config.mode == "train":
        parts = corpus.final_parts(config.split_manifest, config.calibration_fraction, config.split_seed)
        print("Fit budget: 1 model; no automatic hyperparameter search", flush=True)
        train_fit(config, parts, output)
    else:
        development = corpus.poems("train") + corpus.poems("validation")
        folds = plain_folds(development, config.folds, config.split_seed)
        print(f"Fit budget: {len(folds)} models; fixed supplied configuration; validation results", flush=True)
        plan = [{role: [p.poem_id for p in poems] for role, poems in fold.items()} for fold in folds]
        plan_path = output / "folds.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("CV partitions changed; use a new output_dir")
        atomic_json(plan_path, plan)
        results = [
            train_fit(replace(config, mode="train"), dict(fold, calibration=[]), output / f"fold-{n}")
            for n, fold in enumerate(folds, 1)
        ]
        atomic_json(
            output / "cross_validation.json",
            {
                "folds": results,
                "score_role": "Checkpoint-selected validation; not independent test",
                "mean_poem_macro_f1": sum(r["selection_metrics"]["poem"]["macro_f1"] for r in results) / len(results),
            },
        )


if __name__ == "__main__":
    main()
