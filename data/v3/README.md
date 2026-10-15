# Eight-class body-only corpus (v3)

V3 retains every v2 poem and partition, and adds 9,734 explicitly source-annotated
examples of ఆటవెలది, తేటగీతి, కందము and సీసము. Total: **23,029 poems**.
The four original vrutta labels retain their original class IDs (0–3); new labels use 4–7.
V1, v2, their trainers and their released packages remain reproducible.

| Label | Fit | Validation | Reserved calibration | Test | Total |
|---|---:|---:|---:|---:|---:|
| ఉత్పలమాల | 2,215 | 585 | 324 | 996 | 4,120 |
| చంపకమాల | 2,501 | 648 | 276 | 759 | 4,184 |
| మత్తేభము | 1,789 | 423 | 194 | 733 | 3,139 |
| శార్దూలం | 1,070 | 203 | 114 | 465 | 1,852 |
| ఆటవెలది | 289 | 1,092 | 75 | 152 | 1,608 |
| తేటగీతి | 943 | 209 | 10 | 640 | 1,802 |
| కందము | 2,571 | 721 | 435 | 1,071 | 4,798 |
| సీసము | 680 | 250 | 130 | 466 | 1,526 |
| **Total** | **12,058** | **4,131** | **1,558** | **5,282** | **23,029** |

Fit has 73 works, validation 9, reserved calibration 4, and test 13 (99 works total).
Work sizes make the new data allocation differ from a 70/15/15 target; Vemana is a
large single validation work. Never redistribute its lines to improve proportions.
Ataveladi's fitting support is relatively small and validation is concentrated by
work; interpret per-work results alongside pooled metrics. No alternative is better
supported among the explicitly named remaining labels in the original other pool:
Malini 29, Mahasragdhara 23, Sragdhara 16 and Bhujangaprayatamu 2 records before
further screening. Ambiguous `గీ.` is not promoted to Tetageeti automatically.

## Input representation

Every row contains **four ordered full padas**. Ataveladi and Kandamu keep their
unequal odd/even padas. Seesamu uses **only its four body padas**: pair adjacent
half-lines 1+2, 3+4, 5+6, 7+8 from an explicitly marked 8+4 body/closing-geeti
record. The last four display lines are retained in `closing_geeti_lines`, and
all twelve original display lines remain in `raw_lines`. Closure text never
enters the vocabulary or CNN features and is not separately relabelled as training
geeti. It remains part of duplicate/work isolation. The operation restores the
published full-pada grouping; it does not reconstruct missing words or padas.

Only the clear 12-display-line compound layout is admitted in this version.
Six/eight/ten/eleven-line compound records and bodies without an explicit closure
marker remain excluded pending a source-layout audit. This conservative extraction
policy is independent of the body-only prediction contract: inference needs only
the four body padas. A bare body with independently confirmed boundaries and source
label can be added in a future version without requiring its closure as a feature.

## Provenance and isolation

Sources are archived AndhraBharati HTML plus accepted independent and expansion
collections from Wikisource. New labels come from printed `ఆ./ఆ.వె.`, `తే.`, `క.`
and explicit `సీ.+…` markers; no neural/rule predictions or count-based label repairs.
The old scraper discarded non-vrutta cells even when explicitly annotated; this
builder recovers cells with four explicit `<br>`-delimited padas. Latin/OCR/HTML fragments, missing text,
ambiguous layouts and contradictory cross-partition quotations are excluded.
Original report HTML hashes are verified against bytes or the original scraper's
UTF-8/newline-normalized text hashes; preserved raw-byte hashes are recorded anew.

Exact poem deduplication and layout-equivalent line/edition/work grouping apply to
all classes together, including reserved Seesamu closures. Existing v2 fit,
selection, calibration and test roles are pinned before assigning new groups.
The current screen does not establish absence of fuzzy duplicates or paraphrases.
Sources may share authors across roles. Source annotations are not expert gold.
Test contains the already-inspected v2 benchmark and new-label held-out works;
it must not be presented as an entirely fresh blind or unseen-author test.

`manifest.json` hashes the corpus, catalogue and audit. `audit.json` records all
input hashes, raw pools (which overlap and must not be summed as unique examples),
exclusions and policy. [WORKS.md](WORKS.md) / `works.json` list every retained work,
its role, class counts and source URLs. See [SOURCES.md](../SOURCES.md) for rules and
[LICENSE.md](../LICENSE.md) for data rights. The local raw archive is ignored by Git.

## Rebuild, train and predict

The prepared corpus needs no parser dependency. Rebuilding from the local raw
archive requires `uv sync --locked --extra cpu --extra collection` and a **new,
empty output directory**; the builder refuses to overwrite a version.

```bash
python scripts/prepare_eight_class.py --output /tmp/chandassu-v3-rebuild
python train_cnn_expanded.py --config configs/cnn_eight_class.json
python train_cnn_expanded.py --config configs/cnn_eight_class.json --resume
python train_cnn_expanded.py --predict runs/cnn-eight-class/model --input poem.json
```

`poem.json` contains `{"lines": ["pada 1", "pada 2", "pada 3", "pada 4"]}`.
Long Seesamu padas have a separate 512-character encoding capacity; exceeding it
raises an error, never silent truncation. The shared character CNN pools each pada
with masked max/mean, preserves the order of all four vectors, and predicts one of
eight labels with one complete-poem loss. Class weights use square-root inverse
frequency from fit only. One fixed seed/recipe is trained from scratch. Validation
poem Macro F1 selects the checkpoint; unweighted validation poem CE breaks ties.
The test is evaluated only after that checkpoint is frozen. All retained calibration
works are reserved and unused in this experiment; probabilities are uncalibrated.

The package format is `ordered_poem_cnn_v1` with label/vocabulary/config metadata,
checksummed safetensors and a dedicated loader. Use the expanded script for local
prediction. The local server now loads this architecture alongside the original four-class
models; use the server command in the root README. The Hub includes the new standalone eight-class loader at the repository root
and in `cnn-eight-class/`; original packages remain available in their existing
subfolders. ByT5 is parked for this experiment.
A source-labelled raw-text classifier is not a lakshana verifier or rule-distilled
model. Such claims require a validated prosody teacher and appropriate supervision.
