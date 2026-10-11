# Expanded prepared corpus (v2)

V2 includes all v1 poems plus 1,777 poems (7,108 lines) from 34 new works, without duplication.
Of the new poems, 1,058 from 27 works join fitting and 719 from seven works remain held out.
This is the allocation frozen before the completed GX10 trial. All original v1 fitting,
calibration, validation and test poems retain their roles. The validation file is unchanged.
V1 and the published CNN/ByT5 packages remain unchanged.

| Partition | Poems | Lines | Works |
|---|---:|---:|---:|
| Train including calibration | 8,483 | 33,932 | 53 |
| Fit (optimization) | 7,575 | 30,300 | 49 |
| Calibration (inside train) | 908 | 3,632 | 4 |
| Validation / selection | 1,859 | 7,436 | 4 |
| Holdout test | 2,953 | 11,812 | 13 |
| Total | 13,295 | 53,180 | 70 |

The single test file contains all 2,234 original test poems plus 719 new-work poems.
Both cohorts remain excluded from fitting, vocabulary construction, selection and calibration.
They have already been evaluated on earlier models; they are a fixed benchmark, not a fresh blind study.
Fit grows by 1,058 poems relative to v1 and is identical to the completed GX10 trial.
A fresh CNN rerun uses the same fitting examples as that trial; it adds no further poems.

```bash
python train.py --model_type=cnn --config=configs/cnn.json \
  --output_dir=runs/cnn-v2-combined
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
