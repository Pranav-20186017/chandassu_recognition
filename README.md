# Chandassu Recognition

Chandassu Recognition identifies eight Telugu poetic metres:
**ఉత్పలమాల · చంపకమాల · మత్తేభము · శార్దూలం · ఆటవెలది · తేటగీతి · కందము · సీసము**.

The new default is a fresh eight-class character CNN trained on CPU with data v3.
It predicts one label from four ordered full pādas; Seesamu uses its body alone.
Test poem Macro F1 is **94.84%**, with Ataveladi the weakest class at **69.34%**.

The model and standalone loader are published at the
[Hugging Face repository](https://huggingface.co/sprx7767/chandassu-recognition)
root and in `cnn-eight-class/`. The original four-class CNN (NVIDIA GB10/GX10,
data v2) and ByT5 encoder (NVIDIA H100 NVL, data v1) remain in `cnn/` and `byt5/`.
Those original models classify lines and average calibrated probabilities;
the new model uses ordered poem context and reports uncalibrated scores.
ByT5 is parked for the eight-class extension. Ensemble only compatible packages
with the same architecture and label space.

## Eight-class CNN experiment

A separate fresh CNN adds **ఆటవెలది · తేటగీతి · కందము · సీసము** to all four
existing vrutta labels. It uses [data/v3](data/v3/README.md), with 23,029 poems and
frozen work partitions. Seesamu input is its four full body padas; the closing
geeti remains attached as provenance and is excluded from model inputs.

```bash
python train_cnn_expanded.py --config configs/cnn_eight_class.json
python train_cnn_expanded.py --config configs/cnn_eight_class.json --resume
python train_cnn_expanded.py --predict runs/cnn-eight-class/model --input poem.json
```

The new architecture shares a character CNN across padas and predicts once from
all four ordered vectors, using validation poem Macro F1 for checkpoint selection.
It trains from scratch on all eight labels; ByT5 is parked. It has a dedicated
checksummed safetensors package/loader; use the expanded prediction command above.
The original four-class models remain available, and the local
server supports both architectures as separate prediction strategies. See [sources and rules](data/SOURCES.md), [all works](data/v3/WORKS.md),
and [the experiment report](docs/eight_class_experiment.md).

### Serve the eight-class model with the same interface

```bash
python inference.py --model_dirs runs/cnn-eight-class/model \
  models/cnn-seed-17 models/cnn-seed-42 models/cnn-seed-73
```

Open <http://127.0.0.1:8765>. The existing layout, colours, theme toggle and controls
are preserved. The eight-class CNN is the default prediction option; original
single-seed models and their four-class ensemble stay available in the dropdown.
Only compatible original models participate in that ensemble.

The eight-class CNN predicts jointly from four full ordered padas. Seesamu may
also be pasted as eight body half-lines, optionally followed by four closing geeti
lines; the latter are excluded from inference. It reports uncalibrated poem scores
and no per-line predictions. Select an original four-class option for single-pada
inference. To serve only the new model, use
`python inference.py --model_dir runs/cnn-eight-class/model`.

### Load the published eight-class model

The updated [notebook](notebook.ipynb) defaults to the eight-class model and includes
optional original-model examples. It accepts complete poems and Seesamu body input;
its saved outputs are cleared. The standalone loader `hf_poem.py` is included in
the Hub package for Transformers Auto interfaces without installing this project.

```bash
python scripts/load_hf_model.py --poem poem.txt --revision FULL_HF_COMMIT
```

The loading script resolves and pins one HF commit when `--revision` is omitted.
To run the same model through the web interface directly from the Hub:

```bash
python inference.py --model_id sprx7767/chandassu-recognition --revision FULL_HF_COMMIT
```

Use `--subfolder cnn-eight-class` for the explicitly named eight-class package,
or `--subfolder cnn` / `--subfolder byt5` for the preserved original packages.
Only files in the selected manifest are downloaded. See the
[model card and full Auto loading example](docs/huggingface/README.md).

## Install

Install Python 3.12 and [uv](https://docs.astral.sh/uv/).
Run these commands from the project directory:

```bash
uv sync --locked --extra cpu
source .venv/bin/activate
```

For an x86-64 Linux system with CUDA 12.8, use this installation command instead:

```bash
uv sync --locked --extra cuda
source .venv/bin/activate
```

The first ByT5 training run downloads the base model.
CNN training does not require a model download.

## Train a model

Run one of these commands:

```bash
python train.py --model_type=cnn --config=configs/cnn.json --output_dir=runs/cnn-v2-combined
python train.py --model_type=byt5
```

Each command trains one model with the final model hyperparameters.
The CNN preset uses v1 plus the new v2 fitting works and retains the original validation/calibration.
ByT5 uses the original v1 partitions. Both keep all test works out of training and selection.
The script prints the settings before training starts.

### Change the settings

Set the seed, learning rate, batch size, and epoch limit with command-line options:

```bash
python train.py --model_type=cnn --seed=42 --learning_rate=0.0002 \
  --microbatch_poems=8 --effective_batch_poems=16 \
  --epochs=80 --minimum_epochs=24 --output_dir=runs/cnn-seed42

python train.py --model_type=byt5 --microbatch_poems=4 \
  --effective_batch_poems=16 --gradient_checkpointing \
  --output_dir=runs/byt5-checkpointed
```

You can also change [configs/cnn.json](configs/cnn.json) or [configs/byt5.json](configs/byt5.json).
Command-line options override the configuration file.
Use `python train.py --help` to list all options.

Batch sizes count poems.
Each poem supplies four lines.
The microbatch size is the number of poems processed at one time.
The effective batch size is the number of poems used for one weight update.
The effective batch size must be a multiple of the microbatch size.

### Train an ensemble

Train each model separately with its own seed and output directory.
You can also change the hyperparameters for each model:

```bash
python train.py --model_type=cnn --seed=17 --learning_rate=0.0001 \
  --output_dir=runs/cnn-seed-17

python train.py --model_type=cnn --seed=42 --learning_rate=0.0002 \
  --output_dir=runs/cnn-seed-42
```

Use distinct seeds and the same model type for all ensemble members.
Load the completed model packages together:

```bash
python inference.py --model_dirs runs/cnn-seed-17/model runs/cnn-seed-42/model \
  --strategy=ensemble --port=8765
```

### Save and resume training

The script saves these files in the output directory:

| File | Contents |
|---|---|
| `best.pt` | Checkpoint with the highest validation poem Macro F1 |
| `last.pt` | Last completed epoch, weights, optimizer, scheduler, and random states |
| `history.json`, `history.csv` | Loss, F1, learning rate, and duration for each epoch |
| `model/` | Model package for inference |
| `run.json`, `partitions.json` | Training settings and poem assignments |
| `result.json` | Final validation and calibration results |

To resume a run, use the same settings and output directory.
Add `--resume` to the original command:

```bash
python train.py --model_type=cnn --resume
```

Training starts from the last completed epoch.
The script checks that the data, settings, code, and software versions have not changed.

To train an existing model on additional data, use a new output directory:

```bash
python train.py --model_type=cnn --data_dir=/path/to/prepared/data \
  --init_checkpoint=/path/to/stored/models/cnn --output_dir=runs/cnn-more-data
```

This command keeps the model's architecture and input encoding.
It starts a new optimizer.
A CNN maps new characters to its unknown-character token.
Train a new CNN to add characters to its vocabulary.
See [data/README.md](data/README.md) for data preparation instructions.

### Cross-validation

To train three models with separate validation groups, run this command:

```bash
python train.py --model_type=cnn --mode=cross_validate --folds=3 \
  --split_manifest=none --calibration_fraction=0 --split_seed=142 \
  --output_dir=runs/cnn-cv
```

Each fold uses the supplied settings.
The script keeps works together and uses only training and validation data.
This procedure differs from the original experiment.
Its scores are validation scores for the selected checkpoints.

## Data and sources

The poems come from [AndhraBharati](https://andhrabharati.com/) and [Telugu Wikisource](https://te.wikisource.org/).
The combined four-class dataset in `data/v2` contains **13,295 poems** and **53,180 lines** from **70 distinct works**.
V2 already includes v1; do not concatenate them.

| Dataset | Poems | Lines | Works |
|---|---:|---:|---:|
| Training, including calibration | 8,483 | 33,932 | 53 |
| Validation | 1,859 | 7,436 | 4 |
| Holdout test | 2,953 | 11,812 | 13 |
| **Total** | **13,295** | **53,180** | **70** |

The expanded CNN uses **7,575 poems** (**30,300 lines**) to update model weights: all
6,517 original fitting poems plus 1,058 new poems from 27 works. Validation is unchanged.
A separate set of **908 poems**, or **3,632 lines**, adjusts the output probabilities through temperature calibration.
These 908 poems come from the training dataset shown above.

Validation scores determine which checkpoint to save.
Calibration adjusts the probabilities after checkpoint selection.
The single test file contains the original 2,234 test poems plus 719 poems from seven
new works frozen before the GX10 trial. None enter training, vocabulary fitting,
validation or calibration. Both cohorts were previously evaluated on earlier models;
this is a fixed benchmark, not a fresh blind study. The evaluation script reports
the combined score and each cohort separately.

The completed GX10 trial already used this exact fitting/validation/calibration set.
Retraining is a fresh run on the same examples; growth is relative to the v1 baseline.
The selected v2 CNN is seed 73, epoch 58. ByT5 and the original H100 evidence remain unchanged.
See [data/v2/WORKS.md](data/v2/WORKS.md) for the new work assignments.

All four lines of a poem stay in the same dataset.
A work and its related editions stay together.
Works with duplicate lines also stay together, including duplicates with different spacing.
Some authors occur in more than one dataset.

[data/v1/WORKS.md](data/v1/WORKS.md) describes the larger collection of **46 works**.
It includes authors, source links, poem counts, and additional data excluded from the four-class dataset.
Poems from other classes and poems without a known class are excluded from model training.

### Reviewed GX10 CNN release

All three seeds use initial LR 0.0001, 60 completed epochs and the same frozen v2
partitions. Each fits its temperature on the 908 separate calibration poems.

| Seed | Selected epoch | Validation poem Macro F1 | Validation line CE before temperature | Temperature |
|---|---:|---:|---:|---:|
| 17 | 54 | 0.998596 | 0.031031 | 1.339218 |
| 42 | 60 | 0.998596 | 0.031057 | 1.396669 |
| 73 (released) | 58 | 0.998596 | 0.028452 | 1.401755 |

Seed 73 wins the validation F1 tie through lower CE, before test evaluation.
Its checkpoints and export weights, code/runtime identity, split fingerprints,
calibration assignments, and selection rule have been verified. Mac CPU inference
reproduces all test confusion matrices and F1 scores. Local `models/cnn` points to
this seed; all three packages remain in `models/cnn-seed-17`, `models/cnn-seed-42`,
and `models/cnn-seed-73` for backup or optional local ensemble inference.

| Seed 73 benchmark | Poems | Correct | Poem Macro F1 | Line Macro F1 |
|---|---:|---:|---:|---:|
| Original v1 test | 2,234 | 2,234 | 1.000000 | 0.993992 |
| Frozen new-work test | 719 | 716 | 0.996042 | 0.991346 |
| Combined test | 2,953 | 2,950 | 0.999084 | 0.993420 |

Every replica and the equal-probability ensemble make the same three test poem errors.
Matched CPU inference with the previously released v1 seed-17 CNN produces the same
poem predictions. This update shows no poem-level accuracy gain. Combined-test line
Macro F1 changes from 0.992608 (old CNN) to 0.993420 (seed 73); the ensemble reaches
0.993592. Changes in hardware/runtime, vocabulary, seed and data prevent attributing
this difference solely to corpus growth. Validation remains checkpoint-selected;
both test cohorts were inspected on earlier models and are a fixed benchmark.

The complete returned review bundle and verification report are preserved locally
under ignored `runs/cnn-v2-replicates.tar.gz` and `runs/review-gx10-cnn-replicates/`.
The staged Hub CNN package carries the selected model's selection/test/verification reports.
Publishing that package and the prepared model card to the existing Hub repository awaits approval.
Original v1 evidence stays in results/v1; ByT5 is unchanged.

## Model architectures

### Character CNN

![Character CNN architecture](docs/images/cnn-architecture.svg)

The current CNN has **68,484 parameters** with the **93-character vocabulary** built from its fitting poems.
The embedding table adds padding and unknown-character tokens, for 95 tokens in total.
The original H100 v1 CNN had 68,420 parameters and a 92-character vocabulary.
Three convolution branches process each line with different kernel sizes.
Each branch supplies 64 features to the classification layer.

### ByT5 encoder

![ByT5 encoder architecture](docs/images/byt5-architecture.svg)

The ByT5 encoder and classification layer have **217,663,364 parameters**.
The model converts each line to UTF-8 byte tokens.
The encoder produces token features, and the classification layer predicts one of four meters.

The base model is [`google/byt5-small`](https://huggingface.co/google/byt5-small).
The configuration specifies revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`.

### Training hyperparameters

| Setting | CNN | ByT5 |
|---|---:|---:|
| Initial learning rate | 0.0001 | 0.00003 |
| Microbatch size, poems | 16 | 8 |
| Effective batch size, poems | 16 | 16 |
| Evaluation batch size, poems | 16 | 8 |
| Maximum epochs | 60 | 32 |
| Minimum epochs | 24 | 8 |
| Early stopping: epochs without improvement | 12 | 8 |
| Input limit per line | 256 characters | 1,024 byte tokens, including the end token |
| Gradient checkpointing | Off | Off |
| Released seed | 73 | 17 |
| Selected epoch | 58 | 17 |

Both models use the following settings:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Dropout | 0.2 |
| Gradient clipping | 1.0 |
| Learning-rate scheduler | ReduceLROnPlateau |
| Scheduler input | Validation line cross-entropy loss, without class weights |
| Scheduler patience | 4 epochs |
| Learning-rate multiplier | 0.5 |
| Minimum learning rate | 0.000001 |
| Minimum F1 improvement | 0.000001 |
| Minimum loss improvement for an F1 tie | 0.0001 |
| Random seeds | 17, 42, 73 |
| Poem order | Fixed |

The scheduler reduces the learning rate when validation loss stops improving.
Validation poem Macro F1 determines the best checkpoint.
For an F1 tie, lower validation line loss determines the checkpoint.
All training settings are available as command-line options.

## Original H100 results (v1)

[results/v1](results/v1) contains the final results for all three folds and all six final models.
It includes training histories, class scores, work scores, tables, and figures.

| Measure | CNN | ByT5 |
|---|---:|---:|
| Cross-validation poem Macro F1, mean across seeds | 0.997130 | 0.997137 |
| Correct historical test poems, for each seed | 2,234 / 2,234 | 2,233 / 2,234 |
| Historical test poem Macro F1 | 1.000000 | 0.999619 |

The cross-validation row reports scores for poems with meter labels supplied by the sources.
Macro F1 gives equal weight to each of the four classes.
The two models had similar scores within the experiment's comparison margin of 0.002 F1.
The CNN is smaller and costs less to run.

The historical test poems had been examined during earlier project work.
These scores therefore describe performance on known test data.
They do not establish performance on a new, independent test set.
The labels combine source annotations and rule-based checks.
Independent experts have not verified every label.

Both models can assign high probabilities to poems from other meters.
Use them to classify the four listed meters.
The reported probabilities are not statistical confidence intervals.
The models do not produce syllable divisions, Guru/Laghu patterns, or gaṇa sequences.

## Inference

The web interface accepts a single line or a four-line poem.
It displays the predicted meter, class probabilities, and line predictions.
You can select a single model or an ensemble.
You can also switch between light and dark themes.

### Load in Python without cloning this project

Install the tested dependencies in your Python environment:

```bash
pip install "torch==2.8.0" "transformers==4.57.6" "safetensors==0.8.0"
```

Use the Transformers Auto interfaces to load the tokenizer and classifier.
The tokenizer applies the same text normalization and input encoding used in training.
The model returns four raw logits for each line.
Apply the saved calibration temperature, then average line probabilities to classify a poem.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "sprx7767/chandassu-recognition"
options = {"subfolder": "cnn", "trust_remote_code": True}
# Change "cnn" to "byt5" to load the ByT5 encoder classifier.
tokenizer = AutoTokenizer.from_pretrained(repo, **options)
model = AutoModelForSequenceClassification.from_pretrained(repo, **options).eval()

poem = """శ్రీ కైవల్య పదంబుఁ జేరుటకునై చింతించెదన్ లోక ర
క్షైకారంభకు, భక్త పాలన కళా సంరంభకున్, దానవో
ద్రేకస్తంభకుఁ, గేళి లోల విలసద్దృగ్జాల సంభూత నా
నా కంజాత భవాండ కుంభకు, మహానందాంగనాడింభకున్."""
lines = [line.strip() for line in poem.splitlines() if line.strip()]
if len(lines) not in (1, 4):
    raise ValueError("Supply one line or a four-line poem")
inputs = tokenizer(lines, padding=True, return_tensors="pt")
with torch.inference_mode():
    logits = model(**inputs).logits
    line_scores = (logits.double() / model.config.calibration_temperature).softmax(dim=-1)
    scores = line_scores.mean(dim=0)

winner = scores.argmax().item()
print(model.config.id2label[winner], f"{scores[winner].item():.2%}")
for index, score in enumerate(scores):
    print(model.config.id2label[index], f"{score.item():.2%}")
```

`AutoModel.from_pretrained(repo, **options)` also loads this classifier.
`trust_remote_code=True` loads the Python classes supplied in the model repository.
For a fixed model version, add `"revision": "<full commit ID>"` to `options`.
The example was tested with Python 3.12 and the dependency versions listed above.
Input limits are 256 characters per line for CNN and 1,024 byte tokens, including the end token, for ByT5.
The tokenizer rejects overlong input and does not truncate lines.

Class probabilities are calibrated scores, not statistical confidence intervals.
A standard text-classification pipeline applies its own softmax to line logits.
Use the calculation above for calibrated poem scores.
Clone this project when you need the web interface, ensembles, or training.

### Download trained models

Download the final CNN or ByT5 package from [Hugging Face](https://huggingface.co/sprx7767/chandassu-recognition).
Run these commands from the project directory with its Python environment active.

For CNN:

```bash
hf download sprx7767/chandassu-recognition \
  --include 'cnn/*' --local-dir models/hf

python inference.py --model_dir=models/hf/cnn --strategy=single --port=8765
```

For ByT5:

```bash
hf download sprx7767/chandassu-recognition \
  --include 'byt5/*' --local-dir models/hf

python inference.py --model_dir=models/hf/byt5 --strategy=single --port=8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

### Export an original H100 checkpoint

New runs already provide native `model/` packages. The following adapter reads the original H100 bundle.
Replace `/path/to/stored/models` with the directory that contains your saved training bundle.
Export the required model:

```bash
python scripts/export_model.py --bundle /path/to/stored/models \
  --model_type=cnn --seed=17 --output_dir=models/cnn-v1

python scripts/export_model.py --bundle /path/to/stored/models \
  --model_type=byt5 --seed=17 --output_dir=models/byt5
```

These export commands expect the completed training bundle format.
For a new training run, use its `model/` directory directly.
To add Transformers Auto loaders to a local package, run:

```bash
python scripts/export_hf.py --model_dir=/path/to/model/package \
  --output_dir=/path/to/transformers/package
```

This command preserves the weights and adds the tokenizer, classifier code, and Auto mappings.
Use a separate, empty output directory.
The exported package also works with `inference.py`.
For a Hub repository with model subdirectories, use `--hub_dir=/path/to/hub/root` to stage the shared code and model card.

Start the server with a model package:

```bash
python inference.py --model_dir=/path/to/model/package --strategy=single --port=8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).
Use `models/cnn-v1` or `models/byt5` as the model directory for these original H100 exports.
The current GX10 release uses `models/cnn` (seed 73).

### Load an ensemble

Export a separate package for each seed:

```bash
for seed in 17 42 73; do
  python scripts/export_model.py --bundle /path/to/stored/models \
    --model_type=cnn --seed="$seed" --output_dir="models/cnn-seed-$seed"
done
```

Start the server with all three packages:

```bash
python inference.py \
  --model_dirs models/cnn-seed-17 models/cnn-seed-42 models/cnn-seed-73 \
  --strategy=ensemble --port=8765
```

The ensemble calculates the mean of the models' calibrated class probabilities.
Use at least two different seeds from the same model type.
Use `--strategy=single` to make the first package the default model.
The interface can switch between loaded models without a server restart.
The same commands support ByT5 packages.

The displayed score range compares the loaded models.
It is not a confidence interval.
The server evaluates all loaded models to calculate that range.
Load one package to reduce inference time.

### Use the API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Server status |
| `GET /api/info` | Model details |
| `POST /api/predict` | Meter prediction |

Submit a four-line poem in a JSON request:

```bash
curl http://127.0.0.1:8765/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"line one\nline two\nline three\nline four","strategy":"ensemble"}'
```

Use `"single"`, `"ensemble"`, or a loaded seed such as `"seed-42"` for the strategy.
Omit the strategy to use the server default.

The response includes class probabilities, line logits, line agreement, probability margin, entropy, temperature, and inference time.
CNN responses also include the fraction of unknown characters.
The server returns HTTP 422 for unsupported line counts or inputs above the model's length limit.

## Run with Docker

### Inference on CPU

Build the image:

```bash
docker build --target cpu -t chandassu:cpu .
```

Replace `/absolute/path/to/model/package` with the directory of one exported model.
Start the server:

```bash
docker run --rm -p 127.0.0.1:8765:8765 \
  -v /absolute/path/to/model/package:/app/models/current:ro \
  chandassu:cpu python inference.py --host=0.0.0.0 \
  --model_dir=/app/models/current --strategy=single --port=8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

### CNN training on GX10 / GB10

Reuse the H100-selected CNN recipe: initial LR **0.0001**, seeds **17, 42, 73**.
H100 searched three rates using seed 17 and repeated the winning rate at the other
seeds. It did not establish an independently optimal LR per seed. We reuse that fixed
recipe on v2 without another rate search or the earlier nested-fold study.

From a fresh clone, use the existing Dockerfile's GX10 target with Docker/GPU access enabled:

```bash
docker build --platform=linux/arm64 --target=gx10 -t chandassu:gx10 .
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -v "$PWD:/app" -w /app chandassu:gx10
```

This runs `train.py --model_type=cnn --config=configs/cnn.json --mode=replicates`
with output `runs/cnn-v2-replicates`, CUDA/BF16, and three fresh CNNs. All reuse the
same 7,575 fitting, 1,859 validation, 908 calibration, and 2,953 test poems. Each CNN
fits its own temperature on the separate calibration works after checkpoint selection.
The 908 calibration poems are inside canonical train but never update CNN weights.
The scheduler can reduce the initial LR; the fixed recipe is not a rate search.
A normal `--mode=train` command still trains one model.

The run writes `seed-17/`, `seed-42/`, and `seed-73/`, each with all checkpoints,
history, a native inference package and `test.json`. `replicate_plan.json` fixes the
settings/partitions before fitting. After all three finish, `selection.json` records
the single winner using selected-checkpoint validation poem Macro F1, unweighted
validation line CE as a tie-break, and seed as a deterministic final tie-break.
Test scores never choose that winner. All three are then evaluated on the combined
test and its two cohorts. `ensemble-test.json` evaluates a predetermined equal-probability
ensemble of all three separately calibrated packages; it does not optimize membership
or weights on test. The top single model is the Hugging Face candidate. The other
models remain available for backups or optional local ensemble inference.

A complete `runs/cnn-v2-replicates.tar.gz` is created automatically after test evaluation.
It includes every model/checkpoint/report and the exact prepared corpus. On your Mac,
copy it from the fresh clone (adjust the remote repository directory if needed):

```bash
scp -i ~/.ssh/pranav_ed \
  pranav@100.98.23.103:~/chandassu-clean/runs/cnn-v2-replicates.tar.gz \
  /Users/pranav/code/chandassu_recognition/runs/
```

For an interrupted batch of runs, repeat the Docker command with this explicit command:

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -v "$PWD:/app" -w /app chandassu:gx10 \
  python train.py --model_type=cnn --config=configs/cnn.json --mode=replicates \
  --output_dir=runs/cnn-v2-replicates --device=cuda --precision=bf16 --resume
```

Resume verifies completed models, continues an interrupted seed, and starts any seed
not reached yet. Data, code, settings and runtime must match. Test reports are recomputed
after all models are complete, without changing the frozen validation-selected winner.

The target retains NVIDIA PyTorch/CUDA from `nvcr.io/nvidia/pytorch:25.11-py3`, the base
used in the successful GX10 trial. Application dependencies use the pyproject pins in
an isolated environment. USER/LOGNAME and a writable cache address the previous UID error.
The new image has not been built locally because the local Docker daemon is stopped;
the original GPU base was verified by the completed trial. Runtime versions are recorded.

The returned three-seed run has been reviewed: seed 73 is the selected CNN package
for the prepared CNN-only Hub release and model card update. Hub publication is pending;
ByT5 stays unchanged. Future reruns
should verify the package listed in their own selection.json before publishing.
For local ensemble inference after extracting the archive on your Mac:

```bash
python inference.py \
  --model_dirs runs/cnn-v2-replicates/seed-17/model \
               runs/cnn-v2-replicates/seed-42/model \
               runs/cnn-v2-replicates/seed-73/model \
  --strategy=ensemble --port=8765
```

### Training on a CUDA GPU

Use an x86-64 Linux host with an NVIDIA GPU and NVIDIA Container Toolkit.
Build the image:

```bash
docker build --target cuda -t chandassu:cuda .
```

Create the output directory:

```bash
mkdir -p runs
```

Start training:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/runs:/app/runs" -e HF_HOME=/app/runs/huggingface \
  chandassu:cuda python train.py --model_type=byt5 \
  --device=cuda --precision=bf16 --output_dir=/app/runs/byt5
```

Use `--model_type=cnn` to train the CNN.
The container writes checkpoints and the model package to the mounted output directory.

## License and citation

You may use, modify, and share the project code and project-owned dataset annotations under [LICENSE](LICENSE).
**You must cite Chandassu Recognition when you use either the code or the dataset.**

Include the citation in publications and reports.
For other uses, keep the citation in the application or dataset documentation.
Preserve the original poem authors and source links when you share data.

Third-party poems and source editions retain their own terms; see [data/LICENSE.md](data/LICENSE.md).
Earlier Apache-2.0 releases retain their original permissions.

Use this BibTeX entry:

```bibtex
@misc{pranav2026chandassu,
  author = {Pranav Surampudi},
  title = {Chandassu Recognition: Telugu Poetic Meter Classification},
  year = {2026},
  howpublished = {\url{https://github.com/Pranav-20186017/chandassu_recognition}},
  note = {Code, dataset annotations, and trained-model results}
}
```

[CITATION.cff](CITATION.cff) supplies the same citation for GitHub and citation tools.
Record the version or commit used in your work.
