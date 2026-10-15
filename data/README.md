# Prepared poetry data

The eight-class CNN experiment uses [v3/](v3/README.md): all v2 poems plus
Ataveladi, Tetageeti, Kandamu and body-only Seesamu (23,029 poems total).
See [source/rule references](SOURCES.md) and [work catalogues](WORKS.md).
Its dedicated entry point is `train_cnn_expanded.py`; the commands below retain
the frozen four-class workflow.

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

The current four-class expansion is `v2/`, which already includes v1. It has 8,483 train poems
(including 908 calibration), 1,859 unchanged validation poems, and 2,953 test poems.
The fitting partition has 7,575 poems: all original fitting poems plus 1,058 new ones.
The seven new held-out works (719 poems) were frozen before the GX10 trial and join
all 2,234 original test poems. No original fitting poem was removed. See v2/WORKS.md.

```bash
python train.py --model_type=cnn --config=configs/cnn.json --output_dir=runs/cnn-v2-combined
```

Use `v2/splits/train_expansion.json` to preserve the original calibration and validation.
The default CNN preset selects it; ByT5 still uses v1. New data allocations preserve
existing roles and assign only new whole work/edition/duplicate groups. Freeze these
assignments before training, without choosing them using test metrics. Never randomly
split lines or move existing fitting works into validation to grow validation.
Fresh CNN training on the accumulated fitting data is the default; resume applies to
an interrupted run with identical data/settings. V1 stays frozen for reproducibility.

Source URLs describe provenance, not blanket redistribution permission. See LICENSE.md.
