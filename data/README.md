# Prepared poetry data

`v1/manifest.json` verifies the released files. Target JSONL files contain one complete
four-line poem per row, never separate line records. This is a lossless conversion of
the frozen corpus's text, labels, IDs and split roles; inference applies the same
NFC/BOM removal/whitespace normalization as the H100 code.

Required target fields: `poem_id`, `source` (work title), `author`, `lines` (four ordered
nonempty strings), and `label` (one of the four supported Telugu labels).
Metadata retains `source_url`, `edition_family`, `label_status`, `label_origin`, audit
status, correction IDs, original labels and available rule/version provenance.
`source_annotation`, `rule_inferred` and `rule_verified` labels are distinguished.
The underlying detailed audit/HTML evidence stays in the local archive.

`works.json` is the machine-readable catalogue; `WORKS.md` links all source editions.
It includes non-target works so corpus coverage is explicit. Canonical target counts:
7,425 train, 1,859 validation, 2,234 historical-test poems. There are 11,518 target poems
and 46,072 target lines. The larger collection has 16,453 poem records across 46 works;
these collection counts include other/unknown domains and 1,323 target records from
previous historical partitions or quarantine, excluded from the final supervised files.
The catalogue distinguishes retained target counts from excluded records. `other.jsonl` and `unknown.jsonl`
retain role and provenance and are never supplied to four-class optimization.

`splits/final_h100.json` freezes fit/selection/calibration poem IDs from the completed
H100 run. Calibration is part of canonical train, but excluded from optimization.
Source titles in a shared `edition_family`, and works connected by layout-equivalent
lines, must remain together. Author overlap is permitted and recorded; do not claim
unseen-author evaluation.

To train on more data, create a new prepared directory rather than overwrite v1:

```bash
python scripts/prepare_data.py --input=all-new-poems.jsonl \
  --output_dir=data/v2 --split_seed=142
python train.py --model_type=cnn --data_dir=data/v2 \
  --split_manifest=none --calibration_fraction=0.15 --output_dir=runs/cnn-v2
```

The preparation helper allocates whole work/edition/duplicate components in one fixed
seeded allocation. Fractions are fractions of groups, so poem proportions can differ.
It reports class coverage failures without trying seeds repeatedly. Inspect and freeze
work assignments before training; do not optimize split choices using test metrics.
For curated assignments, supply train/validation/test JSONL directly and a schema-2
manifest with file hashes and the exact label order, matching v1. `--calibration_fraction=0`
uses all canonical train poems and reports uncalibrated probabilities. A supplied
`--split_manifest` takes precedence over calibration fraction/seed; omit it for a new split.

Source URLs describe provenance, not blanket redistribution permission. See LICENSE.md.
