# Research archive

The archive preserves the project evidence before consolidation.
The active workflow uses `taining.ipynb` and `data/corpus/v1/`.

`notebooks.zip` contains all nineteen saved notebook files.
This includes NB01 through NB09 and three earlier output snapshots.
It also retains seven automatic notebook snapshots.
It also includes the latest locally saved NB09 outputs.
`notebooks_manifest.json` records each file's original SHA-256.
Regression tests verify every archived byte.
The archive does not execute when training starts.

To extract one notebook into a temporary folder:

```bash
.venv-local/bin/python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
output = Path('/tmp/chandassu-notebook-history')
with ZipFile('archive/notebooks.zip') as archive:
    archive.extract('notebooks/09_audited_direct_comparison.ipynb', output)
print(output)
PY
```

`data/` retains the complete original collection tree.
The CSV files, manifests, cached HTML, and source exports keep their original bytes.
Older manifests record their original paths under `data/`.
The historical readers resolve those paths under `archive/data/`.
The canonical corpus has its own self-contained paths and hashes.

Old notebooks describe the code and paths at their recorded time.
Treat them as evidence.
Use the active notebook for new runs.
The historical test suite reads the notebook archive for regression checks.
Original local model directories remain under `runs/`.
The cleanup does not remove or modify those directories.

`local-run-inputs/` preserves locally generated original split files and a test ledger.
These files were already outside Git.
They remain outside Git after the move.
The canonical release contains the supported complete splits.
