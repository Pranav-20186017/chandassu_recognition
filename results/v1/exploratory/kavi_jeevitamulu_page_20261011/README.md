# Ramaraajabhushana page: frozen external evaluation

Both frozen model families matched all **12 fresh, unambiguous four-line quotations** at poem level, for **100% accuracy and Macro F1 = 1.000**. The CNN also matched all 48 individual lines in this subset in every seed. This is a small, source-annotated external check; the labels have not received independent expert verification.

Source: [కవి జీవితములు / రామరాజభూషణకవి, Telugu Wikisource](https://te.wikisource.org/wiki/కవి_జీవితములు/రామరాజభూషణకవి). Public DOM captured on 2026-10-11. Classical quotations are presented in Gurajada Sriramamurti's biographical text. The saved page retains source attribution and links; see Wikisource for its licensing notices.

## Extraction

Extracted every occurrence explicitly marked **ఉ.**, **చ. / చం.**, **మ.**, or **శా.**. The page uses **చ.** for its two Champakamala quotations.

| Metre | Labelled occurrences | Fresh benchmark poems | Fresh lines |
|---|---:|---:|---:|
| ఉత్పలమాల | 7 | 4 | 16 |
| చంపకమాల | 2 | 2 | 8 |
| మత్తేభము | 7 | 3 | 12 |
| శార్దూలం | 5 | 3 | 12 |
| Total | 21 | 12 | 48 |

The 21 labelled occurrences contain **106 lines**: 19 ordinary four-line occurrences, one 26-line continuous quotation, and one four-line quotation reassembled from fragments explicitly supplied by the source. Two pairs repeat the same verse with transcription variants, so the occurrences represent 19 distinct quotations. The 26-line passage is scored as a whole passage and at line level; it is not divided into invented four-line poems.

Metre prefixes, enclosing quotation marks, footnote links, and bibliographic citations were removed from model input. Page-break continuations were joined. Source spelling was retained; no prediction-dependent text repairs were made. The special fragment reconstruction is fully documented in `selection.json` and excluded from the strict benchmark.

The `ప.` quotation at paragraph 193 lacks a specific metre label. It was also scored for inspection, but excluded from labelled metrics. Both models predict ఉత్పలమాల. All other metre prefixes, including సీ., క., తే., and గీ., are inventoried and excluded rather than assigned target labels by the model.

## Results

Seed 17 is the primary model in each family; seeds 42 and 73 are stability checks. Every model was loaded from the completed RunPod bundle with its saved temperature. Poem probabilities are the arithmetic mean of calibrated line probabilities. Checkpoint hashes and fit identities were verified. No model was trained, selected, recalibrated, or modified for this batch.

| Evaluation subset | Quotations / lines | CNN seed 17 | ByT5 seed 17 | Poem Macro F1, both |
|---|---:|---:|---:|---:|
| Fresh, unambiguous, nonoverlapping four-line quotations | 12 / 48 | 12/12; 48/48 lines | 12/12; 48/48 lines | 1.0000 |
| All ordinary complete four-line occurrences, labels as printed | 19 / 76 | 18/19; 72/76 lines | 18/19; 72/76 lines | 0.9606 |
| All labelled quotations, including the long passage and reconstruction | 21 / 106 | 20/21; 102/106 lines | 20/21; 102/106 lines | 0.9641 |

All three CNN seeds and all three ByT5 seeds have the same poem-level predictions. ByT5 seeds 42 and 73 predict మత్తేభము for the first line of `p122-07`, whose source label is శార్దూలం. Their aggregate prediction for that poem remains శార్దూలం. Their fresh-subset line accuracy is therefore 47/48 (97.92%); each CNN seed and ByT5 seed 17 achieves 48/48.

The fresh-subset confusion matrix is diagonal in class order ఉత్పలమాల, చంపకమాల, మత్తేభము, శార్దూలం, with supports **4, 2, 3, 3**. Per-class precision, recall, and F1 are all 1.000. Detailed matrices and per-class statistics for every subset and seed are in `results.json`.

## The apparent mismatch is a source-label conflict

The verse beginning **నను శ్రీరామపదారవిందభజనానందున్** appears twice:

- `p018-01`: the page marks it **మ.** (మత్తేభము).
- `p156-01`: the page marks its transcription variant **ఉ.** (ఉత్పలమాల).

Every CNN and ByT5 seed predicts **మత్తేభము for both versions**. Against the labels exactly as printed, the latter is recorded as a mismatch. Both occurrences are excluded from the strict benchmark because the page contradicts itself. Their orthographic akṣara counts are 20 per line, consistent with మత్తేభము, but this count alone is not a full metrical proof. The later ఉ. marker is likely a source error; an expert check would establish that rather than accepting the models' agreement as gold.

## Freshness and exclusions

The strict subset was defined before inference using source integrity and overlap checks, not model scores. Nine occurrences were excluded:

- Two conflicting-label occurrences (`p018-01`, `p156-01`).
- Three previously tested occurrences: the first Vasucharitramu poem and both variants of the third poem (`p055-01`, `p087-01`, `p119-01`).
- Two Manucharitramu quotations found in archived corpus provenance (`p124-01`, `p133-01`). One is an exact/layout-equivalent poem; the other is a 96.45% character-similar transcription with an equivalent line.
- The 26-line passage and the reconstructed quotation (`p190-01`, `p194-01`).

No matching lines or poems were found in the current final training, validation, or test JSONL files. The two corpus overlaps above are in provenance with role `historical_original_test`, not current final training. They were conservatively excluded because they are already known project data.

Checks covered all current target/other/unknown roles and the full provenance poem file, using normalized letter/mark hashes that ignore whitespace and punctuation. Near-match screening compares verses sharing their initial 12 letters and similar lengths; it can miss substantially rewritten quotations. It does not prove an entire author's work was absent or that a pretrained ByT5 model never encountered a public poem. Fresh here means no detected poem/line overlap in those checked files and not previously probed in this session.

## Files and reproduction

- The cached page and original extraction script are retained only in the local archive; the public page is linked above.
- `marker_inventory.json`: every detected verse-prefix occurrence.
- `selection.json`: extraction, original labels, overlap evidence, exclusions; frozen before inference.
- `results.json`: all six models' probabilities, logits, line predictions, checkpoint identities, and metrics.
- `predictions.csv`: compact per-quotation comparison.
- `line_disagreements.csv`: every line disagreement, with the source conflict identified.
- `poems.md`: all extracted texts ready to copy, with predictions and seed score spreads.

The original extraction/inference script is locally archived. These recorded probe outputs remain unchanged; the supported inference entry point is now `inference.py`.

Uses the bundle at `~/Downloads/chandassu-models` and the locally cached ByT5 configuration. Scores are conditional on the four classes. A softmax probability or three-seed spread is **not** a statistical confidence interval. This encouraging result supports external recognition of these quotations; 12 poems, including only two Champakamala examples, are too few to establish broad generalization or other-metre rejection.
