---
language: [te]
license: other
license_name: chandassu-citation
license_link: LICENSE
library_name: transformers
pipeline_tag: text-classification
tags: [telugu, poetry, poetic-meter, cnn, eight-class, safetensors]
---

# Chandassu Recognition — eight-class poem CNN

**Author: Pranav Surampudi**

The default model is the new **eight-class ordered poem CNN**, trained from scratch
on the original four vruttas plus ఆటవెలది, తేటగీతి, కందము and సీసము.
**Test poem Macro F1: 94.84% · accuracy: 97.90%** on 5,282 held-out poems.
Ataveladi is the weakest class (69.34% F1); inspect the class-specific results below.
This is a raw-text classifier with uncalibrated scores, not a formal prosody verifier.

## Packages and loader

| Package | Location | Output | Status |
|---|---|---|---|
| New eight-class CNN, seed 17 | Root, or `cnn-eight-class/` | Eight raw logits per complete poem | Current default |
| Original four-class CNN | `cnn/` | Four raw logits per line | Preserved |
| Original ByT5 encoder | `byt5/` | Four raw logits per line | Preserved; no retraining |

`hf_poem.py` supplies the self-contained new tokenizer, config and model classes.
`hf_model.py` and all original `cnn/` and `byt5/` files remain unchanged. The
[original four-class card](legacy-four-class/README.md) retains their detailed
architecture, calibration, results and usage. Do not mix their label spaces in an ensemble.

[Open the updated notebook](notebook.ipynb) or use
[Colab's Hugging Face URL](https://colab.research.google.com/github/Pranav-20186017/chandassu_recognition/blob/main/notebook.ipynb).
It defaults to the new model, pins one resolved revision for all loaders, and
retains an optional original-model example. Published execution outputs are cleared.

## Python loading

Tested with Python 3.12, PyTorch 2.8.0, Transformers 4.57.6, SafeTensors 0.8.0 and
huggingface-hub 0.36.2. The exported loader requires only PyTorch, Transformers and
the standard library; no project checkout, ByT5 download or login is needed.

```bash
pip install "torch==2.8.0" "transformers==4.57.6" "safetensors==0.8.0" "huggingface-hub==0.36.2"
```

```python
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "sprx7767/chandassu-recognition"
revision = HfApi().model_info(repo, token=False).sha  # resolve once; keep this commit
options = {"revision": revision, "trust_remote_code": True, "token": False}
# Optional: options["subfolder"] = "cnn-eight-class"
tokenizer = AutoTokenizer.from_pretrained(repo, **options)
model = AutoModelForSequenceClassification.from_pretrained(repo, **options).eval()

poem = """శ్రీ కైవల్య పదంబుఁ జేరుటకునై చింతించెదన్ లోక ర
క్షైకారంభకు, భక్త పాలన కళా సంరంభకున్, దానవో
ద్రేకస్తంభకుఁ, గేళి లోల విలసద్దృగ్జాల సంభూత నా
నా కంజాత భవాండ కుంభకు, మహానందాంగనాడింభకున్."""
inputs = tokenizer(poem, padding=True, return_tensors="pt")
with torch.inference_mode():
    scores = model(**inputs).logits.double().softmax(-1)[0]
print(model.config.id2label[int(scores.argmax())])
for i in scores.argsort(descending=True).tolist():
    print(model.config.id2label[i], f"{float(scores[i]):.2%}")
```

`trust_remote_code=True` loads the inspectable `hf_poem.py` from the pinned commit.
Use `AutoModel` or `AutoModelForSequenceClassification`; both load the same classifier.
The [loading script](load_hf_model.py) reads a complete UTF-8 poem text file:

```bash
python load_hf_model.py --poem poem.txt --revision FULL_HF_COMMIT
```

### Whole-poem input contract

- Pass one complete newline-separated poem string, or a list of complete poem strings.
- Four full ordered pādas are required for the general eight-class prediction.
- Eight Seesamu body half-lines are paired into four full body pādas. Twelve lines
  mean eight body half-lines plus four closing geeti lines; the closing geeti is excluded.
- An eight-line input means body half-lines, not four full body pādas plus four geeti lines.
- Maximum 512 characters per full pāda; overlong inputs raise rather than truncate.
- Token tensors have shape `[batch, 4, tokens]`; logits have shape `[batch, 8]`.
- Scores are uncalibrated. There are no independent per-line scores or line probabilities to average.
- A standard line-level text-classification pipeline is unsupported; use the Auto interfaces above.

The original four-class models still support one line or a four-line poem with their
original tokenizers and temperature-calibrated line averaging; see the preserved card.

## Architecture and training

The 266,056-parameter CNN shares 64-wide character embeddings and convolution
branches (kernels 3, 5, 7) across all four ordered pādas. Each branch uses masked
max and mean pooling. The four vectors are concatenated in order and passed to a
128-unit classification head. The saved weights are unchanged by export.
No source IDs, labels, rule scores, closing geeti, or intermediate prosody features
enter the model. The vocabulary is fitted only on optimization data.

Fresh seed 17; AdamW at 0.001; weight decay 0.01; batch 32 poems; dropout 0.2;
gradient clip 1; square-root inverse-frequency class weights from fit only.
Up to 40 epochs, minimum 16, early-stopping patience 8. ReduceLROnPlateau monitors
unweighted validation poem CE (patience 3, factor 0.5, floor 1e-6).
Validation **poem Macro F1** selects checkpoints, with unweighted poem CE for ties.
Epoch **11** was selected after **19** completed epochs; validation Macro F1 was
**86.14%**. Training used CPU/fp32; ByT5 was parked. The 1,558 reserved calibration
poems were unused, so scores must not be described as calibrated confidence.

## Data and results

Data v3 contains **23,029 complete poems from 99 works**. It includes every v2
poem with its existing role preserved. Original target classes retain IDs 0–3;
new classes use IDs 4–7 in the order listed below.

| Class | Fit | Validation | Reserved | Test | Test F1 |
|---|---:|---:|---:|---:|---:|
| ఉత్పలమాల | 2,215 | 585 | 324 | 996 | 99.75% |
| చంపకమాల | 2,501 | 648 | 276 | 759 | 99.67% |
| మత్తేభము | 1,789 | 423 | 194 | 733 | 99.80% |
| శార్దూలం | 1,070 | 203 | 114 | 465 | 99.78% |
| ఆటవెలది | 289 | 1,092 | 75 | 152 | 69.34% |
| తేటగీతి | 943 | 209 | 10 | 640 | 92.27% |
| కందము | 2,571 | 721 | 435 | 1,071 | 99.63% |
| సీసము body | 680 | 250 | 130 | 466 | 98.50% |
| **Total** | **12,058** | **4,131** | **1,558** | **5,282** | **Macro F1 94.84%** |

On the same 2,953 original-label v2 test poems, four-label retention Macro F1 is
99.81% (all eight prediction classes still available), versus 99.91% for the
installed original four-class CNN. This diagnostic comparison does not establish
future-source equivalence, and the old classifier cannot recognize the new labels.

Whole poems, works, edition families and exact-duplicate-linked works stay together.
Seesamu closures are included in isolation checks even though they are not features.
The exact-duplicate screen is not proof against fuzzy duplicates or paraphrases.
Authors can overlap across roles. Vemana dominates Ataveladi validation, while its
fit set has only 289 poems across seven works. Source annotations are not expert gold;
the test includes the previously inspected v2 benchmark. It is not wholly blind or
an unseen-author evaluation. No test metric selected the checkpoint, recipe or split.

One pre-test trial was stopped to exclude 23 new records with identifiable source
corruption; it is kept locally as discarded audit evidence. The reported fit uses
the same fixed seed and recipe, with no test-informed tuning.

[Complete reports and figures](https://github.com/Pranav-20186017/chandassu_recognition/tree/main/results/v3) ·
[Dataset](https://github.com/Pranav-20186017/chandassu_recognition/tree/main/data/v3) ·
[Sources and rules](https://github.com/Pranav-20186017/chandassu_recognition/blob/main/data/SOURCES.md) ·
[All works](https://github.com/Pranav-20186017/chandassu_recognition/blob/main/data/v3/WORKS.md)

## Existing web interface

Clone the [project](https://github.com/Pranav-20186017/chandassu_recognition) and install
with `uv sync --locked --extra cpu`. Download and serve the new package:

```bash
hf download sprx7767/chandassu-recognition --include 'cnn-eight-class/*' --local-dir models/hf
python inference.py --model_dir models/hf/cnn-eight-class
```

Open <http://127.0.0.1:8765>. The appearance is preserved. The eight-class option
reports joint poem scores, while original packages can be loaded alongside it as
separate dropdown strategies. Only compatible original models participate in their
legacy ensemble. The local API is `POST /api/predict` with a `text` field.

## License and citation

Project-owned code and weights follow the [Chandassu citation license](LICENSE).
Earlier Apache-2.0 permissions remain in effect; the retained ByT5 base encoder is
Apache-2.0. Poems, editions, transcriptions and annotations retain their source rights;
these have not been independently cleared for every edition. Preserve authors and
source links. See the project's dataset rights statement and [CITATION.cff](CITATION.cff).
