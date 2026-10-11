"""Prepare an explicitly supplied poem JSONL; group works before splitting."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from chandassu import LABELS
from chandassu.data import assert_isolated, holdout, poem_dict, read_poems, summary
from chandassu.io import atomic_bytes, atomic_json, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=142)
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 1 - args.test_fraction or not 0 < args.test_fraction < 1:
        parser.error("Positive validation/test fractions must sum to less than one")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output must be empty; existing split assignments are preserved")
    poems = read_poems(args.input)
    # One fixed allocation. Class coverage failures are reported, never searched away.
    development, test = holdout(poems, args.test_fraction, args.split_seed)
    train, validation = holdout(development, args.validation_fraction / (1 - args.test_fraction), args.split_seed + 1)
    parts = {"train": train, "validation": validation, "test": test}
    assert_isolated(parts)
    files = []
    for role, rows in parts.items():
        name = f"{role}.jsonl"
        atomic_bytes(
            args.output_dir / name,
            "".join(json.dumps(poem_dict(p), ensure_ascii=False, sort_keys=True) + "\n" for p in rows).encode(),
        )
        files.append(name)
    works = defaultdict(list)
    for role, rows in parts.items():
        for poem in rows:
            works[poem.source].append((role, poem))
    atomic_json(
        args.output_dir / "works.json",
        [
            {
                "work": work,
                "authors": sorted({p.author for _, p in entries}),
                "role": entries[0][0],
                "poems": len(entries),
                "lines": 4 * len(entries),
                "source_urls": sorted({p.metadata["source_url"] for _, p in entries if p.metadata.get("source_url")}),
                "rights_status": "Verify independently for each source edition",
            }
            for work, entries in sorted(works.items())
        ],
    )
    files.append("works.json")
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 2,
            "labels": list(LABELS),
            "files": {name: sha256(args.output_dir / name) for name in files},
            "counts": {role: summary(rows) for role, rows in parts.items()},
            "split_seed": args.split_seed,
            "split_unit": "Whole work/edition/duplicate component",
            "fractions_are_work_fractions": True,
            "input_sha256": sha256(args.input),
        },
    )
    print(json.dumps({role: summary(rows) for role, rows in parts.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
