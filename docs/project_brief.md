# Project brief

Recognize four Telugu poetic metres: ఉత్పలమాల, చంపకమాల, మత్తేభము and శార్దూలం.
A labelled poem has exactly four ordered lines. Every split must keep them together;
work/edition groups and duplicate-linked works are held together as well.

## Model A: direct classification, established

The completed H100 study established direct raw-text classification using a character
CNN and a pretrained ByT5-small encoder with a masked-mean classification head.
Line logits are calibrated using separate works and averaged to classify a poem.
Select checkpoints using validation **poem Macro F1**, with unweighted line CE as a
practical tie-break. Historical test data stay outside optimization, selection and
calibration; already-inspected test results are diagnostic rather than a fresh blind test.

The CNN is the preferred deployment family from the recorded development comparison.
Both supported implementations must remain reproducible with explicit seeds, data
partitions, hyperparameters, checkpoints and runtime metadata. Default commands run
one fit. Optional plain work-grouped folds use a fixed supplied configuration and
explicit fit budget; there is no nested CV or automatic rate search in the supported trainer.
CNN replicates reuse one fixed recipe across explicit seeds and the same frozen final
partitions. Rank single models on selected-checkpoint validation poem Macro F1 with
unweighted validation line CE as the tie-break, then evaluate test. Each model fits
its temperature on separate calibration works. Keep every replica for local use;
the validation winner is the CNN-only Hub candidate. A predetermined equal-probability
ensemble may be reported without using test to choose members or mixture weights.
See README.md and results/v1 for the completed architectures, settings and evidence.

## Model B: structured prosody, future work

The intended next research direction is supervised intermediate reasoning:
Telugu text → akṣara segmentation → Guru/Laghu → gaṇa sequence → metre.
A hybrid neural segmenter with a deterministic metre verifier is also a candidate.
This requires trustworthy intermediate labels and an independently validated verifier;
the current classifiers do not produce these explanations. Establish Model A first.

A small decoder/LLM can later generate structured explanations from supervised targets.
Do not imply that an encoder classifier already implements reasoning, or use reinforcement
learning before supervised targets and a reliable verifier exist. Evaluate future work
on genuinely unseen works, noise and independently annotated examples, with source,
label and licensing provenance retained.

## Release requirements

A flat `train.py --model_type=cnn|byt5` interface and `inference.py` server, locked uv
dependencies, CPU/CUDA Docker targets, whole-poem data with a work catalogue, validation
checkpoint selection, restartable training, complete final reports, separate data rights
status and locally preserved research archives. Model weights are distributed through
Hugging Face rather than Git. The completed models were trained on NVIDIA H100 NVL.

## Eight-class Model A CNN extension (October 2026)

The separate `train_cnn_expanded.py` experiment trains a fresh raw-text CNN on all
four established vruttas plus Ataveladi, Tetageeti, Kandamu and Seesamu. ByT5 is
parked for this extension. Data v3 includes every v2 record with unchanged roles.
Keep four full ordered padas per input; Seesamu's body padas are paired from eight
explicit half-lines. Retain its closing geeti as provenance and in duplicate/work
isolation, outside model features. Predict one label per complete ordered poem.

Freeze new work/edition/duplicate group roles before training; never move existing
fitting works to validation. Use validation poem Macro F1 for checkpoint selection
and unweighted validation poem CE for ties. Test never drives fitting, scheduling,
selection, class substitution or hyperparameter changes. Reserved calibration works
are unused in this initial experiment and probabilities are uncalibrated. Preserve
the four-class workflow and packages; use a distinct eight-class architecture ID.
This remains Model A classification. Rule distillation/verification is future work
until trustworthy prosody supervision and an independently validated teacher exist.
