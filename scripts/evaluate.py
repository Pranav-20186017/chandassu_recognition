"""Explicit diagnostic evaluation after training, without updating a model."""

import argparse
from pathlib import Path

import numpy as np

from chandassu import LABEL_TO_ID
from chandassu.data import Corpus, assert_isolated, read_poems
from chandassu.evaluation.metrics import line_poem_metrics, softmax
from chandassu.inference.predictor import Predictor
from chandassu.io import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--model_dir")
    models.add_argument("--model_dirs", nargs="+", help="Equal calibrated-probability ensemble of distinct CNN seeds")
    parser.add_argument("--data_dir", default="data/v1")
    parser.add_argument("--input", help="Optional independent four-line poem JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args(argv)
    corpus = Corpus(args.data_dir)
    development = corpus.poems("train") + corpus.poems("validation")
    poems = read_poems(args.input) if args.input else corpus.poems("test")
    assert_isolated({"development": development, "diagnostic": poems}, require_all_classes=False)
    predictors = [Predictor(folder) for folder in (args.model_dirs or [args.model_dir])]
    if args.model_dirs:
        seeds = [p.metadata.get("seed") for p in predictors]
        if (len(predictors) < 2 or any(p.config["model_type"] != "cnn" for p in predictors)
                or None in seeds or len(set(seeds)) != len(seeds)):
            raise ValueError("Ensemble requires at least two CNN packages with distinct seeds")
    evaluation_ids = {p.poem_id for p in poems}
    for predictor in predictors:
        for partition in predictor.metadata.get("partitions", {}).values():
            if evaluation_ids.intersection(partition["poem_ids"]):
                raise ValueError("Evaluation IDs overlap model development partitions")
    print(f"Evaluating {len(poems):,} complete poems with {len(predictors)} model(s) on CPU", flush=True)
    lines = [t for p in poems for t in p.lines]
    if len(predictors) == 1:
        logits = predictors[0].logits(lines, args.batch_size)
        temperature = predictors[0].config["temperature"]
        model_info = predictors[0].info()
    else:
        probabilities = np.mean([
            softmax(p.logits(lines, args.batch_size), p.config["temperature"]) for p in predictors
        ], axis=0)
        logits, temperature = np.log(np.maximum(probabilities, 1e-300)), 1.0
        model_info = {"strategy": "ensemble", "mixture": "Equal calibrated line probabilities",
                      "members": [p.info() for p in predictors]}
    targets = [LABEL_TO_ID[p.label] for p in poems for _ in p.lines]
    grouped_logits = logits.reshape(-1, 4, 4)
    work_metrics = {}
    for work in sorted({p.source for p in poems}):
        indices = [i for i, poem in enumerate(poems) if poem.source == work]
        work_targets = [LABEL_TO_ID[poems[i].label] for i in indices for _ in range(4)]
        work_metrics[work] = line_poem_metrics(
            grouped_logits[indices].reshape(-1, 4), work_targets, temperature
        )
    cohort_metrics = {}
    cohorts = corpus.manifest.get("test_cohorts", {}) if not args.input else {}
    if cohorts:
        allocated = [pid for cohort in cohorts.values() for pid in cohort["ids"]]
        if len(allocated) != len(set(allocated)) or set(allocated) != evaluation_ids:
            raise ValueError("Test cohorts must allocate every test poem exactly once")
        for name, cohort in cohorts.items():
            ids = set(cohort["ids"])
            indices = [i for i, poem in enumerate(poems) if poem.poem_id in ids]
            ys = [LABEL_TO_ID[poems[i].label] for i in indices for _ in range(4)]
            cohort_metrics[name] = line_poem_metrics(
                grouped_logits[indices].reshape(-1, 4), ys, temperature
            )
    metrics = line_poem_metrics(logits, targets, temperature)
    atomic_json(
        Path(args.output),
        {
            "model": model_info,
            "role": "Independent input" if args.input else "Held-out benchmark; already inspected on earlier models",
            "metrics": metrics,
            "work_metrics": work_metrics,
            "cohort_metrics": cohort_metrics,
        },
    )
    print(f"Test poem Macro F1={metrics['poem']['macro_f1']:.6f}; report={args.output}", flush=True)
    for name, metric in cohort_metrics.items():
        print(f"  {name}: {metric['poem']['observations']:,} poems; "
              f"Macro F1={metric['poem']['macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
