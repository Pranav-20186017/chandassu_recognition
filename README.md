# Chandassu Recognition

Model A: a PyTorch/Hugging Face Transformer classifies each Telugu pādam as
**ఉత్పలమాల, చంపకమాల, మత్తేభము, or శార్దూలం**. A complete poem has four
pādams with one shared label. Poem predictions average the four probability vectors.

## Layout

```text
data/                       # tracked scraped corpus; other local data ignored
notebooks/                  # exploratory notebooks
scrape/                     # source adapters and scraping instructions
src/chandassu/
  data/                     # normalization, validation, leakage-safe splitting
  models/                   # Transformer classifier and tokenized dataset
  prosody/                  # syllable-count screening; future Model B
  training/                 # weighted CE, checkpointing, prediction
  evaluation/               # line/poem metrics and confusion matrices
configs/model_a.json
.python-version             # Python 3.12.10
uv.lock                     # exact dependency resolution
examples/toy_corpus.jsonl    # fabricated pipeline fixture, NOT valid metrical poetry
tests/
docs/
```

## Install

Python 3.12 (pinned to 3.12.10); use uv and the committed `uv.lock` for local
development. On GX10, use the [Docker/Jupyter setup](docs/gx10.md), which keeps
NVIDIA's GB10-compatible PyTorch instead of installing the generic training
extra. A CPU/MPS environment is sufficient for development.

```bash
uv sync --locked --extra train --group dev
uv run --locked --extra train pytest -q
```

Preparation uses only the Python standard library; `uv sync --locked` suffices.

For the source scraper and complete shatakam directory, syllable-count screening, and CSV exports, see [scrape/README.md](scrape/README.md).
Use `uv sync --locked --extra scrape` to install scraping dependencies independently
of training dependencies. `uv.lock` pins transitive packages; `.python-version`
pins the interpreter. Run commands with `uv run --locked` after syncing.
When training, include `--extra train` with `uv run` to retain the training extra.

The default encoder is [ai4bharat/indic-bert](https://huggingface.co/ai4bharat/indic-bert),
a compact ALBERT encoder whose supported languages include Telugu. Its repository
requires accepting access conditions and authenticating with Hugging Face. Use
`hf auth login` after accepting the model's conditions. Change `model_name` to a
local pretrained encoder or another compatible Hugging Face model as needed.
[IndicBERTv2-MLM-only](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only) is an
alternative, but its model card lists 278M parameters, so it is larger than the
compact starting point. Pin `revision` to a commit SHA for reproducible downloads.
The classification head is initialized for the four labels and must be fine-tuned.

## Corpus contract

CSV (UTF-8, header) or JSONL, one object per line:

```json
{"poem_id":"work1:0001","source":"work1","author":"author1","line_no":1,"text":"తెలుగు పాదం","label":"ఉత్పలమాల"}
```

All six fields are required. `poem_id` must be globally unique across works;
`line_no` must be 1–4. Every poem must have exactly four lines, a consistent label,
source, and author. Labels must use the exact names above. Empty source/author is
allowed for poem grouping, but the chosen source/author holdout field cannot be
empty. Use consistent, canonical metadata to prevent accidental group aliases.
For scraped poem-level records, expand the four lines into four records with the
same poem ID before importing. Retain raw scraped text separately for auditing.

Normalization applies NFC, removes BOM, and collapses whitespace. It preserves
vowel length signs, virama, anusvara, visarga, ZWJ/ZWNJ, digits, and punctuation;
it does not strip metadata, fix OCR, transliterate, or infer sandhi. Remove scraper
headers/verse numbers explicitly upstream if they can reveal the label.

```bash
chandassu inspect --input data/corpus.jsonl
chandassu prepare --input data/corpus.jsonl --output data/splits-poem --group-by poem_id
chandassu prepare --input data/corpus.jsonl --output data/splits-source --group-by source
chandassu prepare --input data/corpus.jsonl --output data/splits-author --group-by author
```

Default split fractions are 70/15/15 with seed 42. Allocation is approximately
class-balanced at **group** level; achieved fractions may differ for large groups.
All four lines stay together. Poems sharing an identical normalized line are
linked, and their complete group components also stay together. This can link
multiple sources/authors and make splitting impossible. Such cases fail clearly;
the implementation never falls back to line-level random splitting. Near-duplicate
verses still need corpus-level review.

Preparation writes train/validation/test JSONL and a manifest containing counts,
class coverage warnings, group options, hashes, and training class weights. Three
nonempty splits and all four training classes are required. Evaluation splits can
lack a class under hard holdouts; the manifest warns and Macro F1 still averages
all four classes (undefined precision/recall/F1 is zero). Review this coverage
before interpreting comparisons. Files are verified against manifest hashes when
loaded. Output directories must be empty to protect previous experiments.

## Train, evaluate, predict

```bash
chandassu train --splits data/splits-source --output runs/model-a-source --config configs/model_a.json
chandassu evaluate --checkpoint runs/model-a-source/best --splits data/splits-source --output runs/source-evaluation
chandassu predict --checkpoint runs/model-a-source/best --text 'తెలుగు పాదం'
chandassu predict --checkpoint runs/model-a-source/best --poem-file data/poem.txt
```

`poem.txt` must contain exactly four nonempty lines. Inference uses the training
normalization and saved context length. Long lines are truncated at `max_length`;
inspect token lengths on the real corpus and increase it within the model's
position limit if needed. Probabilities are model outputs, not calibrated confidence.

Training uses class-weighted cross entropy `N / (4 * n_c)` from training lines
only, AdamW, gradient clipping, seeded sampling, and dynamic padding. Set
`class_weighted` false to benchmark unweighted CE. `device: auto` selects CUDA,
then MPS, then CPU. Set `bf16: true` on a CUDA device supporting bfloat16 for GX10.
The test set is used once after selecting the best validation **line Macro F1**
checkpoint; poem Macro F1 is also reported. Test results should not drive repeated
hyperparameter selection. Seeds improve repeatability but do not guarantee bitwise
identical GPU runs.

Outputs:

- `best/`: reloadable Hugging Face model/tokenizer and inference settings.
- `selection.json`, `history.json`: chosen epoch and per-epoch validation metrics.
- `run.json`: config, split manifest, model commit, PyTorch version, device, weights.
- `test_metrics.json`: line and poem accuracy, Macro F1, per-class precision/recall/F1,
  support, and 4×4 confusion matrices (rows true, columns predicted).
- `test_predictions.json`: individual line/poem predictions and probabilities.

Pretrained weights are not bundled. Scraped CSVs, raw pages, audit reports, and snapshots are tracked under
`data/`; reproduce them with the source scraper. `examples/toy_corpus.jsonl` has
invented labels and numeric class cues; use it only to exercise preparation, never
to claim baseline accuracy. Model B, prosodic rules, domain-adaptive pretraining,
and RL are deferred until the real Model A baseline is established.

## Remote Jupyter on GX10

The root [Dockerfile](Dockerfile) starts JupyterLab with CUDA-enabled PyTorch.
See [docs/gx10.md](docs/gx10.md) for build/run commands, GPU verification,
persistent notebooks, and Jupyter over an SSH tunnel.

From the cloned repository on GX10, start Jupyter in detached mode:

```bash
docker run --rm -d --name chandassu-jupyter \
  --gpus all --shm-size=8g \
  -p 127.0.0.1:8888:8888 \
  -v "$PWD:/workspace" \
  -v chandassu-hf-cache:/home/jupyter/.cache/huggingface \
  chandassu-gx10
```

Get the Jupyter token with `docker logs chandassu-jupyter`. The container keeps
running after you disconnect SSH. On your Mac, start the tunnel:

```bash
ssh -i ~/.ssh/pranav_ed -N -L 8888:127.0.0.1:8888 pranav@100.98.23.103
```

Open `http://localhost:8888/lab`, enter the token, and select the
**Chandassu (GX10 CUDA)** kernel. Open
`notebooks/01_preliminary_model_a.ipynb` and run its cells in order, including
the Hugging Face login and gated-model access check before model loading.
For full training, restart the kernel, open
[`notebooks/02_full_model_a.ipynb`](notebooks/02_full_model_a.ipynb), and run its
cells in order. It uses all training lines for 30 epochs, with batch size 8,
BF16 where supported, class weights from the full training split, and a warmup
followed by linear learning-rate decay. Login is reused from the cache.
No image rebuild is needed; `git pull` updates the mounted notebook.

Progress includes batch loss, throughput, ETA, and GPU memory. Each epoch plots
comparable training/validation loss and line/poem Macro-F1, with validation
confusion matrices and per-class scores. Reports, the best inference checkpoint,
and the latest optimizer snapshot persist under `runs/full-model-a-*/`.
Checkpoint selection uses validation **line Macro-F1**. The optional final-test
cell is disabled until model/settings selection is finished. The notebook does
not automatically resume interrupted training.

Stop Jupyter with `docker stop chandassu-jupyter`; the mounted files remain.

The [full-run investigation](docs/model_a_investigation.md) compares the encoder
with character and count-assisted baselines, audits source labels, and records
the limitations of the single-work validation split.

The [project brief](docs/project_brief.md) records the research roadmap.

## Local CPU development notebook

Python 3.12 and uv are enough for the character baselines. The dedicated
`.venv-local/` environment is explicitly ignored by Git. Dependencies are pinned
in `pyproject.toml` and `uv.lock`, including JupyterLab and the local kernel.

```bash
UV_PROJECT_ENVIRONMENT=.venv-local uv sync --locked --extra analysis --extra notebook
.venv-local/bin/python -m ipykernel install --sys-prefix --name chandassu-local --display-name "Chandassu (local CPU)"
UV_PROJECT_ENVIRONMENT=.venv-local uv run --locked --extra analysis --extra notebook jupyter lab --ip=127.0.0.1 --port=8889
```

Open `notebooks/03_character_baselines.ipynb` and select **Chandassu (local CPU)**.
The notebook already contains the first local run's outputs. Restart the kernel
and run all cells to repeat it; each run writes to a new `runs/character-dev-*`
directory. No GPU, Docker, SSH, or Hugging Face token is needed.

The notebook recreates the original source split if missing, verifies the locked
reference in `configs/model_a_source_split.json`, and evaluates three grouped
folds using **only the original training portion**. Original validation and test
examples are excluded from fitting and development scoring. Reports include
per-work/class metrics, U/M discrimination, progress, and confusion plots.

The [first local development results](docs/character_development_results.md)
record the observed scores and the limits of count-assisted evaluation.
