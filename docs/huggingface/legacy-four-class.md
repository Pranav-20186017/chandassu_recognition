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

The current CNN was trained on **NVIDIA GB10 (GX10)** using the expanded v2 corpus.
The ByT5 package remains the original **NVIDIA H100 NVL** model on v1 data.
This repository provides a character CNN and a ByT5 encoder classifier.
Use the CNN when you need a smaller model.

## Model packages

| Model | Directory | Parameters | Seed | Selected epoch |
|---|---|---:|---:|---:|
| Character CNN | [cnn](https://huggingface.co/sprx7767/chandassu-recognition/tree/main/cnn) | 68,484 | 73 | 58 |
| ByT5 encoder | [byt5](https://huggingface.co/sprx7767/chandassu-recognition/tree/main/byt5) | 217,663,364 | 17 | 17 |

Each directory contains SafeTensors weights, configuration, input encoding, calibration settings, and model metadata.
Use `AutoTokenizer` and `AutoModelForSequenceClassification` to load either model in a Python environment.
You can also use the project loader and web interface.

[Project code and instructions](https://github.com/Pranav-20186017/chandassu_recognition)

## Architectures

The CNN encodes 93 characters, plus padding and unknown tokens, with 64-dimensional embeddings.
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
The current CNN uses **13,295 complete poems**, **53,180 lines**, and **70 retained works**
from v2, which already includes v1. ByT5 continues to use v1: 11,518 poems, 46,072 lines,
and 36 retained works. The CNN update does not retrain or replace ByT5.

| Use | CNN v2 poems / lines | ByT5 v1 poems / lines |
|---|---:|---:|
| Weight training | 7,575 / 30,300 | 6,517 / 26,068 |
| Temperature calibration | 908 / 3,632 | 908 / 3,632 |
| Validation | 1,859 / 7,436 | 1,859 / 7,436 |
| Holdout test | 2,953 / 11,812 | 2,234 / 8,936 |
| Total | 13,295 / 53,180 | 11,518 / 46,072 |

All original v1 split assignments are preserved. Of 1,777 new poems from 34 works,
1,058 poems from 27 works join fitting and 719 poems from seven works remain test.
The new-work test allocation was frozen before training. One CNN test file combines
the original 2,234 test poems and the 719 new-work test poems. Validation is unchanged.
The 908 calibration poems are inside canonical train but never update model weights.

Validation poem Macro F1 selects each checkpoint. Selected-checkpoint validation F1,
then unweighted validation line cross-entropy, selects the released single CNN across
seeds. Temperature is fitted separately for each model after checkpoint selection.
Test data are excluded from fitting, vocabulary construction, selection, and calibration.
All four ordered lines, work/edition families, and duplicate-linked works stay together.
Authors may overlap. Source annotations and rule provenance are retained; these are
not independently verified expert gold labels. Both CNN test cohorts were evaluated
on earlier models: this is a fixed benchmark, not a new blind test or unseen-author study.

The [v2 work catalogue](https://github.com/Pranav-20186017/chandassu_recognition/blob/main/data/v2/WORKS.md)
records the new works and their fitting/test assignments. The complete machine-readable
catalogue includes 80 collected works, of which 70 retain supervised four-class poems.
The [v1 catalogue](https://github.com/Pranav-20186017/chandassu_recognition/blob/main/data/v1/WORKS.md)
retains the original collection, including non-target and excluded material.

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
The H100 study selected initial CNN LR 0.0001; the expanded GX10 run reuses it for
seeds 17, 42, and 73 without another rate search or nested-fold study. All three CNNs
share the same frozen partitions and architecture. They start from fresh weights.
This repository contains the validation-selected CNN seed 73 and unchanged ByT5 seed 17.

The exported calibration temperatures are **1.401754510250526** for CNN and **1.1801772437687996** for ByT5.
CNN runtime: Python 3.12.3, PyTorch 2.10.0a0+b558c986e8.nv25.11, CUDA 13.0, NVIDIA GB10, BF16.
Its training checkout was clean at `f7939ccac30d14bab929ed7d9c91aa202c5a9c80`.

## Current CNN results

| CNN seed | Selected epoch | Validation poem Macro F1 | Validation line CE before temperature | Temperature |
|---|---:|---:|---:|---:|
| 17 | 54 | 0.998596 | 0.031031 | 1.339218 |
| 42 | 60 | 0.998596 | 0.031057 | 1.396669 |
| 73 (released) | 58 | 0.998596 | 0.028452 | 1.401755 |

Seed 73 wins the validation F1 tie through lower validation CE. Test results never
select the seed. Every seed completed 60 epochs, followed by separate temperature
calibration on the 908 calibration poems. Validation scores are checkpoint-selected.

| Released CNN seed 73 benchmark | Poems | Correct | Poem Macro F1 | Line Macro F1 |
|---|---:|---:|---:|---:|
| Original v1 test | 2,234 | 2,234 | 1.000000 | 0.993992 |
| Frozen new-work test | 719 | 716 | 0.996042 | 0.991346 |
| Combined test | 2,953 | 2,950 | 0.999084 | 0.993420 |

All three replicas and their predetermined equal-probability ensemble have the same
combined-test poem score and the same three errors. Matched CPU inference with the
previously released v1 CNN seed 17 produces the same poem predictions on these 2,953
poems. There is **no measured poem-level gain** from this update. Combined-test line
Macro F1 changes from 0.992608 (old CNN) to 0.993420 (new CNN); hardware/runtime,
vocabulary, seed, and data differ, so this does not isolate the effect of corpus growth.
The local three-seed ensemble line Macro F1 is 0.993592; it offers no poem-level gain.
Ensemble membership and equal weights were fixed before evaluation.

The returned checkpoints and native SafeTensors were compared tensor-for-tensor;
file manifests, code identity, all split fingerprints, calibration assignments and
checkpoint selection were verified. Mac CPU inference reproduces the test confusion
matrices and F1 scores. The exported Transformers package preserves the selected weights.
See the CNN directory's selection, test and verification reports.

## Original H100 study (v1), retained for comparison

| Original v1 measure | Original CNN | Unchanged ByT5 |
|---|---:|---:|
| Cross-validation poem Macro F1, mean across three seeds | 0.997130 | 0.997137 |
| Correct historical test poems, seed 17 | 2,234 / 2,234 | 2,233 / 2,234 |
| Historical test poem Macro F1, seed 17 | 1.000000 | 0.999619 |

These original study scores do not describe a repeated cross-validation experiment
on v2. The historical test had been examined during earlier project work. Labels
combine source annotations and rule-based checks; experts have not verified every label.
[Original study evidence](https://github.com/Pranav-20186017/chandassu_recognition/tree/main/results/v1)

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
