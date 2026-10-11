# Completed H100 results

Imported from the completed Downloads/chandassu-models bundle. This directory replaces
the previously committed partial reports and includes **all three folds**, seeds 17,
42 and 73 for both CNN and ByT5, and the final report. No model or optimizer weights
are included. Original imported files are byte-for-byte preserved and hashed in manifest.json.

- `fold_metrics.json`: 18 fold/model/seed records (3 × 2 × 3).
- `development_metrics.json`: full grouped OOF/cohort/calibration/ensemble evidence.
- `development_decision.json`: paired comparison and deployment-family decision.
- `identity.json`, `experiment_plan.json`: original code/runtime/settings hashes and partitions.
- `final_report.json`: complete historical diagnostics and final results.
- `cnn/seed-*`, `byt5/seed-*`: final recipes, histories, historical logits, tables and figures.
- `*/search.json`: recorded final validation-only rate search evidence, retained for audit.
- `exploratory/`: earlier external-work quotation checks; selected sources and records,
  without cached HTML or scraping code. These are exploratory probes, not a blind benchmark.
- `release_*_verification.json`: checks of the new portable inference implementation.

The original study used nested folds and rate search. These records describe that
completed historical experiment; the new supported trainer runs a single fit or plain
explicit folds. It does not recreate nested-study OOF results. Original `runs/...`
paths inside imported reports refer to the H100 job, not paths in this checkout.

The CNN and ByT5 source-annotated grouped OOF means were 0.997130 and 0.997137 poem
Macro F1 respectively. Their difference was practically equivalent within the study
margin; CNN was chosen for lower deployment cost. Historical per-seed correct poems:
CNN 2,234/2,234; ByT5 2,233/2,234. The historical set had been inspected in earlier work.
Other-metre rejection failed and independent expert gold was unavailable. Preserve
these limits when reporting or publishing the results.
