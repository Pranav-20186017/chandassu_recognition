# Chandassu Recognition

Chandassu Recognition identifies four Telugu poetic meters:
**ఉత్పలమాల · చంపకమాల · మత్తేభము · శార్దూలం**.

The project provides a character CNN and a ByT5 encoder classifier.
Both models were trained on an **NVIDIA H100 NVL**.
Use the CNN when you need a smaller model.
You can train several models with different seeds and hyperparameters.
You can then use those models together as an ensemble.

Download the final model packages from [Hugging Face](https://huggingface.co/sprx7767/chandassu-recognition).

Each poem has four lines and one meter label.
The model calculates class probabilities for each line.
It then calculates the mean of those probabilities to identify the poem's meter.

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
python train.py --model_type=cnn
python train.py --model_type=byt5
```

Each command trains one model with the final model hyperparameters.
It uses the same training, validation, and calibration poems as the final models.
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
The four-class dataset contains **11,518 poems** and **46,072 lines** from **36 distinct works**.

| Dataset | Poems | Lines | Works |
|---|---:|---:|---:|
| Training | 7,425 | 29,700 | 26 |
| Validation | 1,859 | 7,436 | 4 |
| Historical test | 2,234 | 8,936 | 6 |
| **Total** | **11,518** | **46,072** | **36** |

The final models use **6,517 poems** to update model weights.
These poems contain **26,068 lines**.
A separate set of **908 poems**, or **3,632 lines**, adjusts the output probabilities through temperature calibration.
These 908 poems come from the training dataset shown above.

Validation scores determine which checkpoint to save.
Calibration adjusts the probabilities after checkpoint selection.
The training script does not use the historical test data.

All four lines of a poem stay in the same dataset.
A work and its related editions stay together.
Works with duplicate lines also stay together, including duplicates with different spacing.
Some authors occur in more than one dataset.

[data/v1/WORKS.md](data/v1/WORKS.md) describes the larger collection of **46 works**.
It includes authors, source links, poem counts, and additional data excluded from the four-class dataset.
Poems from other classes and poems without a known class are excluded from model training.

## Model architectures

### Character CNN

![Character CNN architecture](docs/images/cnn-architecture.svg)

The CNN has **68,420 parameters** with the **92-character vocabulary** built from its training poems.
The embedding table adds padding and unknown-character tokens, for 94 tokens in total.
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
| Selected epoch for seed 17 | 59 | 17 |

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

## Results

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

### Export a local checkpoint

Replace `/path/to/stored/models` with the directory that contains your saved training bundle.
Export the required model:

```bash
python scripts/export_model.py --bundle /path/to/stored/models \
  --model_type=cnn --seed=17 --output_dir=models/cnn

python scripts/export_model.py --bundle /path/to/stored/models \
  --model_type=byt5 --seed=17 --output_dir=models/byt5
```

These export commands expect the completed training bundle format.
For a new training run, use its `model/` directory directly.

Start the server with a model package:

```bash
python inference.py --model_dir=/path/to/model/package --strategy=single --port=8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).
Use `models/cnn` or `models/byt5` as the model directory for the exports above.

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
