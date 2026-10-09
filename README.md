# Chandassu Recognition

Model A: a PyTorch/Hugging Face Transformer classifies each Telugu pādam as
**ఉత్పలమాల, చంపకమాల, మత్తేభము, or శార్దూలం**. A complete poem has four
pādams with one shared label. Poem predictions average the four probability vectors.

After the completed NB08 run, see the [results audit](docs/nb08_results_review.md)
and [research-backed next-step blueprint](docs/next_steps_blueprint.md). A separate
[reviewed corpus](data/expanded_poetry/v2_reviewed/README.md) records 11 rule-supported
label corrections, nine whole-poem quarantines and restored inference provenance
for 143 training annotations. NB08's original inputs/results
remain frozen; the reviewed historical test is not a new blind benchmark.

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

The [remaining-error review](docs/character_error_review.md) checks all hybrid
mistakes and the weakest work against cached sources and Chandam, with three
possible source-label problems flagged for manual review.

## Frozen local baseline and final test

In the same local JupyterLab and **Chandassu (local CPU)** kernel, open
`notebooks/04_final_character_baseline.ipynb`. It verifies the frozen settings,
fits the original training portion, and saves a reloadable `model.joblib` with
its vectorizer, classifier, count mapping, configuration, and dependency versions.
It includes a saved-model prediction example and a separate final-test cell.

**Final-test scoring is enabled by default in notebook 04.** Set
`RUN_FINAL_TEST = False` to fit/save without opening the test. When enabled, the
cell scores the saved model once, prints line/poem and per-work metrics plus U/M
discrimination, and generates confusion and per-class/work plots. It never fits
on original validation or test examples, and retains the original labels.

The persistent `data/character-final-test-v1.json` ledger prevents repeated
prediction across notebook runs. Re-running reads the original completed report;
a newly trained artifact is not evaluated again. Preserve that ledger with its
model and reports under `runs/character-final-*` when moving machines. A started
but incomplete evaluation requires inspecting the partial outputs; it is not
silently retried. Keep subsequent tuning on development folds.

Load a saved baseline for prediction:

```python
from chandassu.models.character import load_character, predict_character
model = load_character("runs/character-final-YOUR_TIMESTAMP/model.joblib")
result = predict_character(model, [line1, line2, line3, line4])
```

## Local sequence-model A/B notebook

Open `notebooks/05_sequence_ab_tests.ipynb` in the same JupyterLab and
**Chandassu (local CPU)** kernel. Install the pinned local sequence dependencies
first, then restart the kernel:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local uv sync --locked --extra analysis --extra notebook --extra sequence
.venv-local/bin/python -m ipykernel install --sys-prefix --name chandassu-local --display-name "Chandassu (local CPU)"
UV_PROJECT_ENVIRONMENT=.venv-local uv run --locked --extra analysis --extra notebook --extra sequence jupyter lab --ip=127.0.0.1 --port=8889
```

The notebook compares character TF–IDF with a small character transformer using
sinusoidal positions, an ordered character CNN, a transformer without positions,
and a grapheme transformer. IndicBERT/ALBERT already includes position embeddings;
this experiment tests representation and architecture. No Hugging Face login or
download is needed. It uses Apple MPS automatically when available, otherwise
CPU; set `DEVICE = 'cpu'` to force CPU.

Run cells in order. The original training split must already exist (prepare it
using notebook 03 if needed). Three outer source/duplicate-held-out folds compare
raw and count-assisted line/poem Macro-F1, U/M discrimination, and per-work scores.
Each neural fit selects its epoch using a separate inner source holdout, then
refits on all outer training data before scoring the outer fold. The cap is 30
epochs, with early stopping after at least 8 selection epochs. One seed is the
default; change `SEEDS` to `[42, 123, 2026]` to check initialization sensitivity.
This is a longer local experiment: four arms each perform selection and refitting
on three folds. Progress, curves, token coverage, confusion matrices, reports,
and reloadable fold checkpoints persist in a new `runs/sequence-ab-*` directory.

Notebook 05 never opens the original validation or consumed final test. These
are exploratory development comparisons; keep the notebook 04 artifact and its
benchmark. Optional cells accept fresh unfiltered four-line poems and report
unsupported counts explicitly. They do not silently exclude difficult inputs
or claim a fresh benchmark when no independently labelled new poems are supplied.

The [first sequence comparison and padding audit](docs/sequence_development_results.md)
records the completed local run. Its CNN scores require a corrected rerun:
the original MPS run learned a nonzero padding embedding, making some predictions
depend on batch padding. Notebook 05 now masks padding explicitly, records
implementation version 2, and checks padding weights during training. Restart
the kernel and run the clean notebook with the same settings first. The original
executed notebook is preserved under `notebooks/results/`.

## Standalone CNN and independent poetry notebook

Open `notebooks/06_independent_cnn_poetry.ipynb` in the same local JupyterLab
and **Chandassu (local CPU)** kernel. Restart the kernel and run its cells in
order; notebook 05 does not need to be open or run first. It uses the same pinned
`analysis`, `notebook`, and `sequence` extras. No Hugging Face download is needed.

The full CNN architecture is defined in notebook cells, with an exported diagram
and parameter summary: character embeddings, explicit padding masks, parallel
3/5/7-character convolutions, ReLU, masked max pooling, concatenation, dropout,
and a four-class head. It trains all 24,516 original training lines for 12 fixed
epochs, chosen as the median of notebook 05's inner-selection epochs [12, 9, 17].
Fresh poetry does not select epochs or hyperparameters. Training progress,
loss/F1 plots, reload checks, and a frozen character reference are included.

The complete fit persists under `runs/cnn-independent-v1/` and is reused on
reruns. Notebook 06 has its own saved model and requires no notebook 05 fold
checkpoints. Original validation and final-test examples are not fitted or scored.

**The separate independent collection is available** at
`data/independent_poetry/v1/poems.csv`: 2,061 target poems, 2,045 other-metre
poems and 23 unlabelled poems from ఆముక్తమాల్యద, పాండురంగమాహాత్మ్యము,
కళాపూర్ణోదయము and ఆంధ్ర పురాణము. All 30 declared chapter/section pages were
collected. Each work also has its own CSV under `v1/works/`. These records are
never merged into training. Notebook 06 already reads this path; restart the
kernel and run its cells in order to fit/reuse the frozen model and evaluate.

Labels are **source annotations**, not independently verified ground truth.
The collection keeps 193 target poems with diagnostic count mismatches,
including 161 from the noisier కళాపూర్ణోదయము transcription. See
`v1/annotation_review.jsonl` before interpreting errors as model failures.
Malformed boundaries, compound stanzas, four existing-corpus overlaps and one
duplicate are quarantined in `v1/review.jsonl`. Neither count agreement nor model
predictions selected the exported poems. The original HTML/rendered DOM,
revision URLs, extraction evidence and novelty report are saved in this directory.

The notebook audits novelty against the full existing corpus solely for overlap
detection, retains count mismatches and unsupported inputs, and distinguishes
source annotations from independently verified labels. Known other metres and
unlabelled poems remain visible. Reported raw/count-assisted line and poem F1,
U/M scores, per-work results, coverage, confusion matrices, and a separately
assessed heuristic rejection rule avoid hiding difficult examples. Unavailable
predictions count as errors in all-target metrics. The rejection threshold is
fixed before test collection and is not calibrated.

When a ready novel batch is supplied, a local ledger freezes dataset/model/config
hashes before scoring. Repeated runs read the completed report; changes or a
partial evaluation block reuse. Preserve the ignored ledger with its referenced
model/report artifacts when moving machines. After inspecting the new set, use
another untouched set for further model selection. See
[`scrape/independent/README.md`](scrape/independent/README.md) for input and annotation rules.

## Longer CNN training and full classification reports

**A fresh ten-work test collection (v2) is now available**, separately from the
NB06/NB07 test batch: **1,874 four-class poems** across 47 original-text pages.
Use `data/independent_poetry/v2/target_poems.csv` for the four target classes,
or `v2/poems.csv` for targets plus 1,501 annotated other-metre poems and 43
unlabelled poems. All four lines remain together and ordered. Source labels are
provisional; 117 diagnostic count disagreements remain included and flagged.
No exact/layout overlap was found with training or v1. Some authors recur, so
this is a new-work holdout rather than an entirely unseen-author benchmark.
**It has not been scored or added to training**, and the notebooks still use v1.
See [`v2 collection notes`](data/independent_poetry/v2/README.md) for the ten works,
per-work CSVs, saved source HTML, revisions, extraction issues and novelty checks.

Rebuild v2 from its committed cache with both required overlap checks:

```bash
uv run --locked --extra scrape python -m scrape.independent.collect_v2
```

Open [`notebooks/07_longer_cnn_training.ipynb`](notebooks/07_longer_cnn_training.ipynb)
in the same **Chandassu (local CPU)** kernel, then Restart Kernel → Run All.
No new dependencies or Hugging Face login are needed. CPU, Apple MPS and CUDA
are supported. The complete CNN and training loop are in notebook cells.

NB07 fits only the locked original 6,129 training poems, and uses the existing
1,463-poem validation split (one held-out work, ఉత్తరరామాయణము) for checkpoint
selection. The separately scraped NB06 collection remains test only; the old
original final test is never opened. Training, validation and test poems are
not mixed. Each batch holds complete four-line poems in line order, with no
shuffling or character permutation. NB06 shuffled line examples between batches,
which did not change character order; NB07 explicitly preserves whole-poem batches.

The fixed 12-epoch control is one shared prefix. The longer arm restores its exact
epoch-12 weights, AdamW moments, RNG states and loader generators, then continues
at epoch 13. It does not independently replay the first 12 epochs and compare
GPU weight hashes. The longer arm runs at least
24 and at most 60 epochs, with validation-loss learning-rate reduction and
early stopping. Select by validation poem Macro-F1, breaking ties with lower
validation line cross-entropy. The saved selected checkpoint is evaluated,
even if it comes from an earlier epoch. Configuration and checkpoint hashes
protect reuse under `runs/cnn-long-v2/`; incomplete or changed runs require
inspection rather than silent retraining. This notebook does not overwrite
NB06's model, reports, dataset or ledger.

The first version's separately trained MPS arms differed numerically despite the
same seed, triggering a strict bitwise comparison before any test evaluation.
Those completed artifacts remain in `runs/cnn-long-v1/`; its executed notebook
is archived under `notebooks/results/07_longer_cnn_mps_20261009_before_shared_prefix.ipynb`.
Restart Kernel → Run All once with the updated notebook/configuration to create
the optimizer/RNG snapshot that version 1 did not save. Version 2 still verifies
that the longer arm restores the actual saved control checkpoint exactly.

Reports include weighted optimization loss, comparable unweighted training and
validation losses, learning curves, accuracy, balanced accuracy, precision/recall/F1
(macro/micro/weighted), per-class specificity, MCC, kappa, confusion matrices,
log loss, multiclass Brier, reliability/ECE, ROC-AUC and average precision.
Both line and poem metrics, per-work metrics, U/M diagnostics, rejection coverage,
other-metre false acceptance and paired comparisons are exported. Probability
metrics explicitly identify covered inputs; all-target classification reports
still count unavailable predictions as errors. No test-based calibration is fitted.

Once checkpoint selection is frozen, NB07 scores the NB06 collection and saves
its own ledger. A verified local NB06 prediction snapshot, when present, adds
the historical baseline without re-running it. This is an **exploratory follow-up
on an already inspected benchmark**, not another blind independent test.
A work-cluster bootstrap uses the same paired poems; only four test works limit
its uncertainty resolution. Do not use these test results for another round of
hyperparameter selection. See the metric-definition references in the notebook.

### NB08: expanded corpus with a new work-level split

Open [`notebooks/08_expanded_corpus_training.ipynb`](notebooks/08_expanded_corpus_training.ipynb)
in the existing local kernel, then **Restart Kernel → Run All**. No new dependencies
or Hugging Face login are needed. The dataset is committed and ready to use.

| Split | Poems | Lines | Works |
|---|---:|---:|---:|
| Training | 7,433 | 29,732 | 26 |
| Validation | 1,859 | 7,436 | 4 |
| Test | 2,235 | 8,940 | 6 |

NB08 intentionally adds two works from NB06 and three from the latest v2 batch
to the original training allocation. Three v2 works join validation. Two NB06
works and four v2 works remain testing only. The two రాధికాసాంత్వనము editions
share individual lines and are kept together in validation. No poem, work or
layout-equivalent line crosses splits; all four lines remain intact and batches
are not shuffled. The original 1,314-poem final test remains reserved.

This supersedes the earlier “v2 has not been added to training” statement **for
NB08 only**. NB06/07 and the source collections retain their historical allocations.
The retained NB06 test cohort was already evaluated; the retained v2 cohort was
fresh at allocation. Report these separately as well as combined. See the
[complete allocation and rebuild notes](data/expanded_poetry/v1/README.md).

The same character CNN trains from scratch for 24–60 epochs, using validation
poem Macro-F1 for selection, validation CE for ties, LR reduction and early
stopping. Atomic last-epoch checkpoints support interrupted training; complete
runs and test predictions are reused under `runs/cnn-expanded-v1/`. The notebook
contains the neural architecture, batch/epoch progress, comparable training and
validation losses, learning curves, per-class and per-work reports, and line/poem
metrics on train/validation/test. Exports cover accuracy, balanced accuracy,
precision/recall/F1, specificity, MCC/kappa, confusion matrices, ROC/PR, log loss,
Brier and reliability/ECE. No scores are fabricated before running the notebook.

## NB09: audited direct-model comparison

Run [NB09](notebooks/09_audited_direct_comparison.ipynb) with **Run All**. It is a
standalone experiment: model definitions and training loops are visible in cells,
with no dependency on previous notebooks/checkpoints. It compares the established
raw-character CNN with a pinned public ByT5 encoder, using three seeds and three
work/component folds, inner validation checkpoint selection, calibration and
other-metre rejection, learning curves, complete line/poem metrics and graphs.
All four lines stay together; no poems, lines or characters are shuffled. Rule
scores never become neural input features or new automatic labels.

```bash
UV_PROJECT_ENVIRONMENT=.venv-local uv sync --locked \
  --extra train --extra sequence --extra scrape --extra analysis --extra notebook --group dev
.venv-local/bin/python -m jupyterlab
```

Default budget is **46 fits**: 18 matched comparison fits, 27 additional CNN
learning-curve fits and one final fit. Allow substantial runtime and disk space,
especially on a laptop. Completed fits are reused, interruptions resume from the
last completed epoch, and changing code/settings/data requires a new run directory.
Reports, checkpoints, predictions and figures go to `runs/nb09-direct-v1/`.
First run downloads ByT5; it is public and does not require gated IndicBERT access.
Optional Hugging Face authentication is provided without saving tokens in reports.

The [frozen audit](docs/analysis/nb09_corpus_audit/README.md) covers 16,453 poems;
16,428 complete inputs received strict and metre-only scans (32,856 requests),
with full responses and input hashes retained. Twelve high-scoring conflicts
remain review candidates. Existing supervised allocation stays at **7,425 train,
1,859 validation and 2,234 historical test poems**. One duplicate quotation in a
development other-metre poem is excluded from threshold calibration, with its
original record retained separately. See the [NB09 guide](docs/nb09.md).

These are grouped development comparisons and historical diagnostics. No new
expert-labelled blind holdout exists yet; NB09 completes normally and reports
that limitation. It keeps the established CNN pending expert-gold confirmation
rather than claiming verified superiority from source annotations alone. A
header-only [fresh-holdout template](data/independent_poetry/nb09_fresh_test_template.csv)
is supplied for genuinely new, independently verified works/authors.
