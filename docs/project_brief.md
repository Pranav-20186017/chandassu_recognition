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
