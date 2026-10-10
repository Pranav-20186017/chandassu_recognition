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
├── configs/experiment.json       # Complete bounded CNN/ByT5 comparison
├── configs/training.json         # Standalone and historical stage settings
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
The default workflow reads `configs/experiment.json` and the canonical corpus.
See [the research protocol](docs/experiment.md) for its peer-reviewed sources and decision rules.

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
The notebook is prepared for the complete comparison with `STAGE = 'end-to-end'`.
Set `STAGE = 'cnn'` when you need to run the CNN baseline.
Run all cells.
The workflow selects MPS when CUDA is unavailable.
It prints progress and saves losses, metrics, predictions, and charts.

The complete study has at most 46 fits, with a fixed candidate search and three matched seeds.
It reuses the measured passing ByT5 check when its identity matches.
It fits final models from both families and evaluates seed and hybrid ensembles.
Outputs use `runs/cnn-byt5-experiment-v2/`.
The report can favour either family, show practical equivalence, or remain inconclusive.
The `cnn` and `byt5-check` stages remain available for standalone runs.
The `nb09-full` stage preserves the previous protocol for historical compatibility.
See [the run guide](docs/training.md) for restart rules and stage details.

For command-line execution:

```bash
.venv-local/bin/python -m chandassu.training --stage end-to-end
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

## Run on RunPod or tune CUDA execution

Use [the RunPod guide](docs/runpod.md) for H100, RTX 4090, and other CUDA Pods.
It covers installation, pinned-model preflight, timing charts, and the notebook or detached CLI run.
The setup preserves the template's CUDA PyTorch instead of installing the Mac build.

```bash
bash docker/setup-runpod.sh
```

Measure microbatch size and checkpointing before the full 46-fit study.
The exported CUDA configuration preserves 16 poems per optimizer update and all study data roles.
It selects execution settings from training-text throughput, not validation or test scores.
No cloud completion time or cost is guaranteed before measurement.

On GX10, use its existing container and run the same benchmark and profile-export commands with `python`.
Do not run the RunPod environment installer in the GX10 container.
Open a new output directory after code, configuration, GPU, or backend changes.
Saved notebook outputs and existing `runs/` directories remain intact.

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
