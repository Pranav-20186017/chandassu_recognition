# Chandassu Recognition — Project Brief

## Goal

Build a Telugu poetic metre recognition system for a restricted four-class problem:

- ఉత్పలమాల
- చంపకమాల
- మత్తేభము
- శార్దూలం

Each poem contains 4 lines/pādams and has one metre label.

The long-term goal is not just to classify the metre statistically, but to build a model that can eventually reproduce the actual chandassu deduction process:

Telugu text\
→ syllabification / akṣara segmentation\
→ Guru/Laghu determination\
→ gaṇa pattern\
→ final metre classification

---

## Dataset

I have scraped a large corpus from Telugu kāvyas and itihāsas.

For a poem:

- 4 pādams
- 1 metre label

Possible line-level training representation:

```text
poem_id | line_id | text | label
```

One poem therefore yields four line-level observations.

### Important split rule

Do **not** randomly split individual lines.

All 4 lines from one poem must remain in the same split to prevent leakage.

Minimum grouping:

```text
group = poem_id
```

Preferably also evaluate harder generalization splits such as:

- unseen source work
- unseen kāvya
- unseen author

The final test set should ideally contain works/authors absent from training.

---

# Model A — Direct Transformer Classifier

This is the first implementation.

## Task

```text
Telugu pādam
→ Transformer encoder
→ classification head
→ one of 4 metres
```

Recommended implementation stack:

- PyTorch
- Hugging Face Transformers
- Telugu/Indic-capable encoder model
- 4-class classification head

IndicBERT-style encoder is a strong starting point.

Conceptually:

```text
Tokenizer
   ↓
Transformer Encoder
   ↓
[CLS] / pooled representation
   ↓
Linear(hidden_dim → 4)
   ↓
Softmax
```

Loss:

```text
CrossEntropyLoss
```

### Class imbalance

If class frequencies differ substantially, use class-weighted cross entropy.

For class \(c\):

\[
w_c = \frac{N}{K n_c}
\]

where:

- \(N\) = total training observations
- \(K = 4\)
- \(n_c\) = observations in class \(c\)

PyTorch:

```python
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
```

Do not rely on overall accuracy alone.

Primary evaluation metric:

```text
Macro F1
```

Also report:

- accuracy
- per-class precision
- per-class recall
- per-class F1
- confusion matrix

Model selection/checkpointing should preferably use validation Macro F1.

Moderate oversampling can be tested later if a minority class has poor recall, but do not aggressively duplicate rare poems.

---

# Model B — Structured Chandassu Reasoning Model

The more interesting long-term architecture.

Instead of:

```text
verse → metre
```

train:

```text
verse
→ syllable / akṣara decomposition
→ Guru/Laghu sequence
→ gaṇas
→ metre
```

This should be more robust to lexical shortcuts and more faithful to the real process of identifying metre.

## Possible architecture

Small Transformer trained specifically for Telugu prosody.

Potential size:

```text
30M–100M parameters
```

Possible encoder-decoder configuration:

```text
d_model: 384–512
layers: 6–12
attention heads: 6–8
context length: 256–512
```

A from-scratch model is feasible because the domain is narrow.

---

## Tokenization

Generic BPE/subword tokenization may not be ideal for metre recognition.

Prefer representations aligned with Telugu phonology.

Possible levels:

```text
Unicode codepoint
→ grapheme cluster
→ akṣara / poetic syllable
```

A Telugu-aware tokenizer should preserve vowel length and consonantal structure required for Guru/Laghu reasoning.

---

## Multi-task training

Instead of only predicting the final metre, train intermediate tasks.

Potential losses:

\[
L =
\lambda_1 L_{\text{syllable}}
+
\lambda_2 L_{\text{guru/laghu}}
+
\lambda_3 L_{\text{gana}}
+
\lambda_4 L_{\text{metre}}
\]

Possible heads:

```text
Transformer representation
 ├─ syllable boundary head
 ├─ Guru/Laghu head
 ├─ gaṇa head
 └─ metre head
```

Class weighting may be applied mainly to the final metre loss.

Intermediate Guru/Laghu supervision is less affected by metre-level class imbalance because all metres provide many Guru/Laghu examples.

---

# Hybrid Architecture

Potentially the most principled system:

```text
Neural model:
Telugu → syllables → Guru/Laghu

Deterministic rule engine:
Guru/Laghu → gaṇas → metre
```

Advantages:

- class imbalance affects the model far less
- deterministic metre rules remain exact
- neural network handles the fuzzy linguistic part
- easier to debug
- easier to interpret

Later, an LLM-style decoder can produce explanations:

```text
Syllables:
...

Guru/Laghu:
...

Gaṇas:
...

Therefore:
శార్దూలం
```

---

# Supervised Reasoning Before RL

RL is not required initially.

The first structured model should use supervised intermediate targets.

Example training output:

```text
Input:
<Telugu pādam>

Target:
Syllables: ...
Guru/Laghu: G L G ...
Gaṇas: ...
Metre: మత్తేభము
```

Once a deterministic verifier exists, reinforcement learning can be added.

---

# Verifier-Based RL

A chandassu system is suitable for rule-based/verifiable rewards.

Possible reward components:

\[
R =
0.25R_{\text{syllable}}
+
0.35R_{\text{guru/laghu}}
+
0.20R_{\text{gana}}
+
0.20R_{\text{metre}}
\]

The verifier can automatically check:

- syllable decomposition
- Guru/Laghu assignments
- gaṇa sequence
- final metre

Recommended progression:

```text
pretraining
→ supervised structured fine-tuning
→ verifier-based RL
```

Do not attempt RL from random initialization.

---

# Domain-Adaptive Pretraining

The large scraped corpus can be used even when metre labels are absent.

Possible pipeline:

```text
pretrained Indic encoder
→ continued masked-language-model pretraining
  on Telugu classical poetry
→ chandassu fine-tuning
```

This can adapt the model to:

- classical vocabulary
- archaic morphology
- poetic syntax
- orthographic conventions
- sandhi
- recurring metrical language patterns

This should be benchmarked against direct fine-tuning.

---

# Experimental Plan

Compare these systems:

| Model | Input | Purpose |
|---|---|---|
| Rule-based | Telugu → prosody rules | deterministic baseline |
| Char CNN / BiLSTM | characters | classical neural baseline |
| Transformer encoder | raw Telugu | Model A |
| Transformer + domain pretraining | raw Telugu | improved Model A |
| Byte/character Transformer | raw Telugu | tokenizer-independent baseline |
| Multi-task Transformer | Telugu + prosodic targets | Model B |
| Neural Guru/Laghu + symbolic rules | Telugu | hybrid system |
| Small decoder/LLM | Telugu → structured explanation | reasoning model |

---

# Key Research Question

Compare:

```text
Model A:
verse → metre
```

against:

```text
Model B:
verse
→ Guru/Laghu
→ gaṇa
→ metre
```

Especially evaluate on:

- unseen authors
- unseen kāvyas
- unusual vocabulary
- synthetic but metrically valid verses
- noisy OCR text

Hypothesis:

Model A may learn lexical/statistical shortcuts.

Model B should generalize better if it truly learns Telugu prosody.

---

# Hardware

Primary training machine:

```text
ASUS Ascent GX10
NVIDIA GB10 / Blackwell
128 GB unified memory
CUDA-capable
```

This should be the main compute environment.

MacBook Air M4 can be used for:

- coding
- preprocessing
- notebook editing
- lightweight debugging

The GX10 should handle:

- full encoder fine-tuning
- from-scratch 30–100M models
- encoder-decoder experiments
- larger batches
- CUDA-specific tooling

---

# Remote Jupyter Setup

The GX10 is available over Tailscale.

Run Jupyter only on loopback:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

On the Mac:

```bash
ssh -N -L 8888:127.0.0.1:8888 USER@GX10_TAILSCALE_HOST
```

Then open locally:

```text
http://127.0.0.1:8888
```

Do not bind Jupyter to:

```text
0.0.0.0
```

unless there is a specific need.

Recommended to run Jupyter inside `tmux`:

```bash
tmux new -s jupyter
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Detach:

```text
Ctrl+B
D
```

Reattach:

```bash
tmux attach -t jupyter
```

---

# Immediate Next Steps

Implement Model A first.

Suggested order:

1. Normalize scraped Telugu text.
2. Create dataset with:
   - `poem_id`
   - `source`
   - `author`
   - `line_no`
   - `text`
   - `label`
3. Inspect class counts.
4. Create grouped train/validation/test splits.
5. Build PyTorch/Hugging Face dataset.
6. Fine-tune a small Telugu/Indic Transformer.
7. Add class-weighted cross entropy.
8. Track Macro F1.
9. Generate confusion matrix and per-class metrics.
10. Establish baseline accuracy before Model B.

After Model A is stable:

11. Implement Telugu akṣara segmentation.
12. Implement deterministic Guru/Laghu rules.
13. Build labelled intermediate prosody targets.
14. Train Model B.
15. Compare Model A vs Model B on unseen-source generalization.
