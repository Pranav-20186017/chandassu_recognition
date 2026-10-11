"""Explicit diagnostic evaluation after training, without updating a model."""

import argparse
from pathlib import Path

from chandassu import LABEL_TO_ID
from chandassu.data import Corpus, assert_isolated, read_poems
from chandassu.evaluation.metrics import line_poem_metrics
from chandassu.inference.predictor import Predictor
from chandassu.io import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", default="data/v1")
    parser.add_argument("--input", help="Optional independent four-line poem JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args(argv)
    corpus = Corpus(args.data_dir)
    development = corpus.poems("train") + corpus.poems("validation")
    poems = read_poems(args.input) if args.input else corpus.poems("test")
    assert_isolated({"development": development, "diagnostic": poems}, require_all_classes=False)
    predictor = Predictor(args.model_dir)
    evaluation_ids = {p.poem_id for p in poems}
    for partition in predictor.metadata.get("partitions", {}).values():
        if evaluation_ids.intersection(partition["poem_ids"]):
            raise ValueError("Evaluation IDs overlap model development partitions")
    logits = predictor.logits([t for p in poems for t in p.lines], args.batch_size)
    targets = [LABEL_TO_ID[p.label] for p in poems for _ in p.lines]
    grouped_logits = logits.reshape(-1, 4, 4)
    work_metrics = {}
    for work in sorted({p.source for p in poems}):
        indices = [i for i, poem in enumerate(poems) if poem.source == work]
        work_targets = [LABEL_TO_ID[poems[i].label] for i in indices for _ in range(4)]
        work_metrics[work] = line_poem_metrics(
            grouped_logits[indices].reshape(-1, 4), work_targets, predictor.config["temperature"]
        )
    atomic_json(
        Path(args.output),
        {
            "model": predictor.info(),
            "role": "Independent input" if args.input else "Historical diagnostic; already inspected",
            "metrics": line_poem_metrics(logits, targets, predictor.config["temperature"]),
            "work_metrics": work_metrics,
        },
    )


if __name__ == "__main__":
    main()
