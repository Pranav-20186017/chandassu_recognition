# Eight-class CNN experiment

The extension keeps the original four vruttas and adds Ataveladi, Tetageeti,
Kandamu and Seesamu. Seesamu is classified from its body alone: eight explicit
half-lines become four ordered full padas, with the closing geeti stored only as
provenance. The [rule/source notes](../data/SOURCES.md) explain the representation.

## Why train a new model

The original CNN learns four-class line predictions and averages their
probabilities. It can technically be adapted by replacing its head and fine-tuning,
but it has no ordered poem-level head. The new experiment trains a modest
266,056-parameter CNN from fresh initialization on **all eight labels together**.
This is one multiclass model. All the requested types can be represented as four
ordered full padas; varying display-line counts do not themselves require a new
model. The original character encoder already accepts variable text lengths.
The new head instead addresses the eight-label output and ordered poem context. Replaying all original fitting poems is essential to retain the
vruttas. There is no weight transfer, ByT5 run, intermediate prosody teacher,
knowledge distillation, or claimed rule verification in this experiment.

A shared character CNN has width 64, kernels 3/5/7, masked max/mean pooling and
0.2 dropout. Concatenating the four pada vectors in source order preserves the
odd/even distinction relevant to Ataveladi and Kandamu. A 128-unit head predicts
one of eight labels with one loss per poem. Labels, sources, rules and closing
geeti are not model features. Padding contributes to neither pool. Character
vocabulary is fitted only on optimization data; unknown characters map to UNK;
512-character line capacity raises on overflow rather than truncating.

## Data and frozen evaluation protocol

[Data v3](../data/v3/README.md) contains 23,029 poems from 99 works: 12,058 fit,
4,131 selection, 1,558 reserved calibration and 5,282 test. It includes every
v2 poem without changing its role. Entire work/edition/exact-duplicate groups
are isolated, including Seesamu closure lines. All eight classes are present in
fit, selection and test. Exact poem deduplication removes repeated records;
conflicting quotations are excluded. This screen does not establish absence of
fuzzy duplicates or paraphrases. Source annotations are not independent expert gold.

| New class | Fit poems / works | Validation poems / works | Test poems / works | Total including reserve |
|---|---:|---:|---:|---:|
| Ataveladi | 289 / 7 | 1,092 / 4 | 152 / 8 | 1,608 |
| Tetageeti | 943 / 9 | 209 / 3 | 640 / 7 | 1,802 |
| Kandamu | 2,571 / 25 | 721 / 8 | 1,071 / 9 | 4,798 |
| Seesamu body | 680 / 11 | 250 / 3 | 466 / 7 | 1,526 |

These are usable supports for an initial experiment; retain all four. Ataveladi
has the least fitting data and Vemana dominates its validation count. More diverse
Ataveladi fitting works would help. Replacing it with Malini (29), Mahasragdhara
(23), Sragdhara (16) or Bhujangaprayatamu (2) from the original explicitly named
other pool would reduce support. Ambiguous `గీ.` is not a substitute class.
See [all works](../data/v3/WORKS.md) and [the audit](../data/v3/audit.json).

The single predeclared recipe uses seed 17, AdamW at 0.001, weight decay 0.01,
batch 32 poems, square-root inverse-frequency class weights derived only from fit,
gradient clip 1, up to 40 epochs, minimum 16, early-stopping patience 8. Reduce LR
by 0.5 after validation CE plateaus for patience 3, floor 1e-6. Select by validation
**poem Macro F1**, then unweighted validation poem CE. Validation is used repeatedly
for checkpoint selection, so its reported score is not an independent test estimate.

Test inference happens after the selected checkpoint/package/result is written.
The original 2,953 v2 benchmark poems were already inspected in earlier work;
new-label test poems also come from known source collections. Report historical
and new-label cohorts and original-four-label retention separately; do not call
the combined evaluation wholly blind or unseen-author. No test score chooses
hyperparameters, labels, split allocations or checkpoints. Reserved calibration
works are unused; predicted probabilities are uncalibrated.

## Run and artifacts

```bash
python train_cnn_expanded.py --config configs/cnn_eight_class.json
python train_cnn_expanded.py --config configs/cnn_eight_class.json --resume
python train_cnn_expanded.py --predict runs/cnn-eight-class/model --input poem.json
```

Configuration: [cnn_eight_class.json](../configs/cnn_eight_class.json).
Artifacts live in `runs/cnn-eight-class/` (ignored by Git): `run.json` binds settings,
data, implementation and runtime; `partitions.json` records exact IDs; `history.json`
records each epoch; `last.pt` saves optimizer/scheduler/loader/random state for an
identity-checked epoch resume; `best.pt` is selected solely on validation.
`model/` contains checksummed safetensors, vocabulary and eight-label metadata.
`result.json` records selected validation performance; `test.json` has class/work
metrics and cohorts, and `test_predictions.json` records every held-out prediction.

The expanded architecture uses its own loader and prediction command. The local
server also loads it as an additional prediction strategy, retaining the original
four-class model and ensemble choices. The new Hub loader is exported as the root default and `cnn-eight-class/`,
while original four-class packages remain in their existing subfolders.

A uniform source-quality screen excludes 23 additional new-label records with
identifiable Latin/OCR/HTML fragments. The interrupted pre-test trial is preserved
locally in `runs/cnn-eight-class-discarded-source-artifacts/` and is excluded from the
reported final model comparison. No new-model test metrics were inspected before
that correction; the fixed architecture, seed, optimizer and selection rule remain
unchanged. The complete v2 records remain frozen.

## Completed result

The fixed seed-17 fit stopped after 19 epochs and selected epoch 11 by validation
poem Macro F1 (86.1444%). On 5,282 held-out poems, test Macro F1 is **94.8425%**
and accuracy is **97.8985%**. Per-class test F1: Ataveladi **69.34%**, Tetageeti
**92.27%**, Kandamu **99.63%**, Seesamu body **98.50%**. The remaining weakness is
Ataveladi/Tetageeti confusion; more diverse Ataveladi fitting data is the clearest
next data improvement, using a future independently frozen evaluation protocol.
The held-out results above must not become tuning feedback for another run on
these same test works.

On the same 2,953 original-label poems, the expanded CNN retains **99.8064%**
four-label Macro F1, versus **99.9084%** for the installed four-class CNN. The
expanded model still has all eight possible predictions in that comparison.
This retention result does not establish equivalence across future sources.

The final model has 266,056 parameters and trained on CPU/fp32 for about 12.7
minutes of epoch time. [Complete reports and figures](../results/v3/README.md)
include the class/work metrics, frozen partitions, recipe, checksums and history.
Local weights and the dedicated inference package are in `runs/cnn-eight-class/`.
The original four-class weights remain available; the expanded model is an initial
research baseline with uncalibrated probabilities and source-annotation limitations.
