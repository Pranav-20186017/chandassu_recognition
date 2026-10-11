# Fresh unseen-work probes — 11 October 2026

Two poems from వసుచరిత్రము were selected before inference from a published transcription with explicit metre markers. The work is absent from all current target, other and unknown splits and from archived provenance. Both poems also have zero complete-line or complete-poem matches after the layout-insensitive hashing used in the project. This excludes exact/layout duplicates; it does not prove semantic independence.

Source: [కవి జీవితములు/రామరాజభూషణకవి](https://te.wikisource.org/wiki/కవి_జీవితములు/రామరాజభూషణకవి). The first poem is marked శా.; the third is marked మ. These are published source annotations, not newly reviewed expert labels. The transcribed text is preserved, with the training normalization applied only during inference.

**ప్రథమాశ్వాసము, పద్యం 1 — reference శార్దూలం**

```text
శ్రీభూపుత్రి వివాహవేళ నిజమంజీరాగ్రరత్నస్వలీ
లాభివ్యక్తి వరాంఘ్రిరేణుభవక న్యాలీల యంచున్ మదిన్
దా భావింప శుభక్రమాకలనచేఁ దద్రత్నముం గప్పుసీ
తా భామాపతి బ్రోవుతన్ దిరుమలేంద్రశ్రీమహారాయనిన్.
```

CNN seed 17: శార్దూలం, 99.46% calibrated score. Three-seed range for this class: 99.42–99.59%.

ByT5 seed 17: శార్దూలం, 99.59% calibrated score. All three CNN seeds and all three ByT5 seeds match the reference.

**ప్రథమాశ్వాసము, పద్యం 3 — reference మత్తేభము**

```text
సకలామోదకతాళవృత్తగతులన్ సంగీతసాహిత్యనా
మకవిద్యాయుగళంబు పల్కుఁజెలికిం బాలిండ్ల జో డైసిరుల్
ప్రకటింపన్ నఖరేఖలందు నలఘుప్రస్తారము ల్సేయుస
ర్వకళాకాంతుఁడు బ్రోవుతన్ దిరుమలేంద్రశ్రీమహారాయనిన్.
```

CNN seed 17: మత్తేభము, 99.71% calibrated score. Three-seed range for this class: 98.47–99.72%.

ByT5 seed 17: మత్తేభము, 99.47% calibrated score. All three CNN seeds and all three ByT5 seeds match the reference.

These two selected poems are a useful fresh probe, not an accuracy benchmark. Once observed, these examples cannot serve as an untouched future test set. No models were retrained, no labels entered model inputs, and no source text was added to the corpus. Raw outputs and checkpoints hashes are in `results.json`; the pre-inference candidate list is in `probes.json`.
