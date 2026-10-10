# Chandassu recognition

Train a four-class Telugu metre classifier from raw poem text.
Open [taining.ipynb](taining.ipynb) for the supported workflow and experiment history.
Python 3.12 is required.

The classes are ఉత్పలమాల, చంపకమాల, మత్తేభము, and శార్దూలం.
Each poem has four ordered lines.
Complete works remain in one split.
Validation poem Macro F1 selects the checkpoint.
Historical test scores do not select model settings.

## Repository layout

```text
chandassu_recognition/
├── taining.ipynb                 # The single active notebook
├── configs/training.json         # Current training and NB09 settings
├── data/corpus/v1/
│   ├── train/                    # Target lines and poems; other and unknown files
│   ├── validation/
│   ├── test/                     # Historical test works
│   ├── provenance/
│   ├── review/
│   └── manifest.json
├── src/chandassu/
│   ├── data/                     # Corpus validation and source readers
│   ├── models/                   # Raw character CNN and raw byte encoder
│   ├── training/                 # Training, restart state, and study stages
│   └── evaluation/               # Metrics, calibration, rejection, and plots
├── tools/
│   ├── collection/               # Source collection tools
│   ├── research/                 # Historical audit and migration tools
│   └── benchmark/                # Hardware throughput checks
├── tests/
├── docs/                         # Run guide, research evidence, and history
├── archive/                      # Original notebooks, source exports, and HTML
├── docker/
└── runs/                         # Local outputs; excluded from Git
```

The training workflow does not read HTML or require a scraping service.
The archive preserves source evidence and notebook outputs.
Historical configuration files remain under `configs/` for regression tests.
The active workflow reads only `configs/training.json`.

## Canonical dataset

| Split | Poems | Lines | Works |
|---|---:|---:|---:|
| Training | 7,425 | 29,700 | 26 |
| Validation | 1,859 | 7,436 | 4 |
| Historical test | 2,234 | 8,936 | 6 |
| Total | 11,518 | 46,072 | 36 |

Labels and text match the reviewed dataset.
The cleanup does not change split membership.
Other metres and unknown labels remain separate from supervised training.
Source annotations and rule-supported labels are not independent expert gold.
See [the dataset guide](data/corpus/v1/README.md) for file definitions.

## Run on the Mac

From the repository root, create or update the local environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local uv sync --locked \
  --extra train --extra sequence --extra analysis --extra notebook --extra scrape --group dev
.venv-local/bin/python -m ipykernel install --user \
  --name chandassu-local --display-name "Chandassu (Python 3.12)"
.venv-local/bin/python -m jupyterlab --ip=127.0.0.1 --port=8889 --no-browser
```

Open `taining.ipynb`.
Select the project kernel.
The notebook is prepared for the longer ByT5 learning check with `STAGE = 'byt5-check'`.
Set `STAGE = 'cnn'` when you need to run the CNN baseline.
Run all cells.
The workflow selects MPS when CUDA is unavailable.
It prints progress and saves losses, metrics, predictions, and charts.

The other stages are `byt5-check` and `nb09-full`.
The full NB09 study has 46 fits.
Run the small ByT5 learning check before the full study.
The check uses 200 updates on the same 16 training poems and saves restart state every five updates.
Its outputs use `runs/training-byt5-check-v2/`; the previous 60-update reports remain in `v1/`.
See [the run guide](docs/training.md) for restart rules and stage details.

For command-line execution:

```bash
.venv-local/bin/python -m chandassu.training --stage cnn
```

## Run on GX10

Build the container on GX10:

```bash
docker build --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" -t chandassu-gx10 .
```

Start Jupyter in detached mode:

```bash
docker run --rm -d --name chandassu-jupyter \
  --gpus all --shm-size=8g \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD:/workspace" \
  -v chandassu-hf-cache:/home/jupyter/.cache/huggingface \
  chandassu-gx10
```

The repository is a volume mount.
The image does not contain project code, data, or HTML.
The image keeps NVIDIA's CUDA-compatible PyTorch build.
The notebook selects CUDA when it is available.
Use the GPU check if another workload consumes unified memory:

```bash
docker exec chandassu-jupyter python docker/check_gpu.py
```

From the Mac, start the SSH tunnel:

```bash
ssh -i ~/.ssh/pranav_ed -N \
  -L 8888:127.0.0.1:8888 pranav@100.98.23.103
```

Read the Jupyter token on GX10:

```bash
docker logs chandassu-jupyter
```

Open `http://127.0.0.1:8888` on the Mac.
Select the installed GX10 kernel if Jupyter requests a kernel.
See [the GX10 guide](docs/gx10.md) for GPU setup and memory diagnostics.

## Verification and evidence

Run the test suite:

```bash
.venv-local/bin/pytest -q
```

[Experiment history](docs/history/experiments.md) describes NB01 through NB09.
[The archive guide](archive/README.md) explains how to recover original notebooks.
[The project brief](docs/project_brief.md) records the original research plan.
The current implementation remains a direct raw-text classifier.
Structured prosody models remain a separate future experiment.
