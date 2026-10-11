# Expanded prepared training data (v2)

Adds 1,777 poems (7,108 lines) from 34 distinct new works to v1 training.
The new poems are entirely in fit. V1 validation and historical test files are byte-for-byte unchanged;
the 908 calibration poems and all existing fit/selection IDs retain their original roles.
The original frozen H100 dataset remains in `../v1`; existing model results describe v1.

| Partition | Poems | Lines |
|---|---:|---:|
| Canonical train | 9,202 | 36,808 |
| Fit (optimization) | 8,294 | 33,176 |
| Calibration (inside canonical train) | 908 | 3,632 |
| Validation / selection | 1,859 | 7,436 |
| Historical test | 2,234 | 8,936 |

```bash
python train.py --model_type=cnn --data_dir=data/v2 \
  --split_manifest=splits/train_expansion.json --output_dir=runs/cnn-v2
```

Use the supplied split manifest to preserve calibration roles. Every row is one complete ordered
four-line poem. New labels come exclusively from printed ఉ, చ/చం, మ, శా annotations.
Annotations and digital transcriptions are not independently verified expert gold.
No model inference, automatic metre inference, OCR, missing-line reconstruction or ellipsis expansion was used.

Novelty checks cover 77,488 existing archived/prepared text records (22,084 unique text keys),
including historical exclusions. They reject known work re-editions and author aliases,
letter/combining-mark-equivalent poems, existing-line overlap and near-duplicate poems
(SequenceMatcher ratio at least 0.90, with 10-character anchor candidate retrieval).
This is an automated conservative screen and author/work review, not proof against every possible textual variant.

[WORKS.md](WORKS.md) lists every new contribution; [EXPANSION.json](EXPANSION.json) records counts and acquisition/audit policy.
Source URLs, revision permalinks and saved HTML hashes are retained in the new poem metadata.
Printed abbreviations and unsupported layouts stay out of training. Six Pandavodyogam poems come
from an indexed anthology; the other new works are available Wikisource transcriptions.

The full local collection is `archive/data/expansion_20261011/`: raw HTML, full document text CSV,
per-work poem/review/target CSVs, all other/unknown text, duplicate decisions, scripts and reports.
It is ignored by Git. Only this prepared target dataset and its necessary partition/provenance metadata are published.
Third-party text and annotation rights follow their sources; see [dataset rights](../LICENSE.md).
