---
language:
  - te
license: other
license_name: chandassu-citation
license_link: https://huggingface.co/sprx7767/chandassu-recognition/blob/main/cnn/LICENSE
library_name: transformers
pipeline_tag: text-classification
tags:
  - telugu
  - poetry
  - poetic-meter
  - cnn
  - byt5
  - safetensors
---

# Chandassu Recognition

**Author: Pranav Surampudi**

Chandassu Recognition identifies four Telugu poetic meters:
**ఉత్పలమాల · చంపకమాల · మత్తేభము · శార్దూలం**.

The final models were trained on an **NVIDIA H100 NVL**.
This repository provides a character CNN and a ByT5 encoder classifier.
Use the CNN when you need a smaller model.

## Model packages

| Model | Directory | Parameters | Seed | Selected epoch |
|---|---|---:|---:|---:|
| Character CNN | [cnn](https://huggingface.co/sprx7767/chandassu-recognition/tree/main/cnn) | 68,420 | 17 | 59 |
| ByT5 encoder | [byt5](https://huggingface.co/sprx7767/chandassu-recognition/tree/main/byt5) | 217,663,364 | 17 | 17 |

Each directory contains SafeTensors weights, configuration, input encoding, calibration settings, and model metadata.
Use `AutoTokenizer` and `AutoModelForSequenceClassification` to load either model in a Python environment.
You can also use the project loader and web interface.

[Project code and instructions](https://github.com/Pranav-20186017/chandassu_recognition)

## Architectures

The CNN encodes 92 characters, plus padding and unknown tokens, with 64-dimensional embeddings.
Three convolution branches use kernel sizes 3, 5, and 7.
Each branch applies ReLU and masked max pooling to produce 64 features.
A linear layer maps the combined 192 features to four class logits.

The ByT5 model uses the encoder from [google/byt5-small](https://huggingface.co/google/byt5-small).
The base revision is `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`.
The encoder has 12 layers and a hidden width of 1,472.
Masked mean pooling and a linear classification layer produce four class logits.

Both models classify individual lines.
Temperature calibration adjusts the line probabilities.
The mean of four line probability vectors gives the poem probabilities.
The class with the highest probability is the poem prediction.

[Architecture diagrams](https://github.com/Pranav-20186017/chandassu_recognition#model-architectures)

## Data and sources

The poems come from [AndhraBharati](https://andhrabharati.com/) and [Telugu Wikisource](https://te.wikisource.org/).
The four-class dataset contains **11,518 poems** and **46,072 lines** from **36 distinct works**.

| Use | Poems | Lines |
|---|---:|---:|
| Weight training | 6,517 | 26,068 |
| Temperature calibration | 908 | 3,632 |
| Validation | 1,859 | 7,436 |
| Historical test | 2,234 | 8,936 |
| **Total** | **11,518** | **46,072** |

Validation poem Macro F1 determines the best checkpoint.
Calibration adjusts probabilities after checkpoint selection.
Historical test data are excluded from training, checkpoint selection, and calibration.
All four lines of a poem stay together.
Works, related editions, and works linked by duplicate lines stay in the same dataset.
Some authors occur in more than one dataset.

The [work catalogue](https://github.com/Pranav-20186017/chandassu_recognition/blob/main/data/v1/WORKS.md) describes the larger collection of 46 works.
It includes authors, source links, counts, and additional data excluded from the four-class dataset.
Poems from other classes and poems without a known class are excluded from model training.

## Training hyperparameters

| Hyperparameter | CNN | ByT5 |
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

Both models use AdamW, weight decay 0.01, dropout 0.2, and gradient clipping 1.0.
ReduceLROnPlateau uses validation line cross-entropy loss without class weights.
The scheduler has patience 4, multiplier 0.5, and minimum learning rate 0.000001.
The experiment used seeds 17, 42, and 73.
This repository contains the final seed-17 package for each model type.

The exported calibration temperatures are **1.34592281013039** for CNN and **1.1801772437687996** for ByT5.

## Results

| Measure | CNN | ByT5 |
|---|---:|---:|
| Cross-validation poem Macro F1, mean across three seeds | 0.997130 | 0.997137 |
| Correct historical test poems, seed 17 | 2,234 / 2,234 | 2,233 / 2,234 |
| Historical test poem Macro F1, seed 17 | 1.000000 | 0.999619 |

The cross-validation row reports scores for poems with source-supplied meter labels.
The historical test poems had been examined during earlier project work.
These scores describe performance on known test data.
They do not establish performance on a new, independent test set.
The labels combine source annotations and rule-based checks.
Independent experts have not verified every label.

[Complete results](https://github.com/Pranav-20186017/chandassu_recognition/tree/main/results/v1)

## Run inference

### Python

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

### Web interface

Install the project and its dependencies:

```bash
git clone https://github.com/Pranav-20186017/chandassu_recognition.git
cd chandassu_recognition
uv sync --locked --extra cpu
source .venv/bin/activate
```

Download and serve the CNN:

```bash
hf download sprx7767/chandassu-recognition \
  --include 'cnn/*' --local-dir models/hf

python inference.py --model_dir=models/hf/cnn --strategy=single --port=8765
```

For ByT5, use these commands instead:

```bash
hf download sprx7767/chandassu-recognition \
  --include 'byt5/*' --local-dir models/hf

python inference.py --model_dir=models/hf/byt5 --strategy=single --port=8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).
Paste a single line or a four-line poem.
The interface displays the meter, class probabilities, and line predictions.
It supports light and dark themes.
The API endpoint is `POST /api/predict` with a JSON request containing `text`.

You can also train additional models with different seeds and hyperparameters.
Use `--model_dirs` and `--strategy=ensemble` to load them together.
Each ensemble requires distinct seeds from one model family.

## Limitations

Use these models to classify the four listed meters.
They can assign high probabilities to poems from other meters.
A high probability does not establish that an input belongs to a supported class.
Class probabilities and score ranges are not statistical confidence intervals.
The models do not produce syllable divisions, Guru/Laghu patterns, or gaṇa sequences.
Performance on new works, different spellings, and damaged text can differ from the reported results.

## License and citation

Project-owned material uses the [Chandassu citation license](https://huggingface.co/sprx7767/chandassu-recognition/blob/main/cnn/LICENSE).
Cite Chandassu Recognition when you use the covered code or dataset material.
Earlier Apache-2.0 permissions remain in effect.
The ByT5 base encoder retains its Apache-2.0 license.

Poems, editions, transcriptions, and source annotations retain their original rights and terms.
Source rights have not been independently verified for every edition.
Preserve poem authors and source links when you share data.
See the [dataset rights statement](https://github.com/Pranav-20186017/chandassu_recognition/blob/main/data/LICENSE.md).

```bibtex
@misc{pranav2026chandassu,
  author = {Pranav Surampudi},
  title = {Chandassu Recognition: Telugu Poetic Meter Classification},
  year = {2026},
  howpublished = {\url{https://github.com/Pranav-20186017/chandassu_recognition}},
  note = {Code, dataset annotations, and trained-model results}
}
```

[Citation metadata](https://huggingface.co/sprx7767/chandassu-recognition/blob/main/cnn/CITATION.cff)
