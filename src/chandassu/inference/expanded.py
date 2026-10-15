"""Serve the ordered eight-class CNN alongside unchanged four-class predictors."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..data import normalize_text
from ..expanded import collate, load_model
from .collection import ModelCollection


class ExpandedPredictor:
    def __init__(self, folder, device="cpu", cpu_threads=4):
        torch.set_num_threads(cpu_threads)
        self.model, self.vocabulary, saved = load_model(folder)
        self.model.to(device).eval()
        self.config = dict(saved, model_type="cnn")
        self.labels = saved["labels"]
        self.device = device
        self.lock = threading.Lock()
        self.metadata = {}
        # Exported packages carry their run reports inside the verified manifest.
        folder = Path(folder)
        metadata_path = folder / "metadata.json"
        if metadata_path.is_file():
            exported = json.loads(metadata_path.read_text())
            self.metadata.update(
                {
                    name: report
                    for name, report in exported.items()
                    if isinstance(report, dict) and report.get("run_id") == saved["run_id"]
                }
            )
        # Native run folders retain the same optional display reports alongside model/.
        for filename in ("run.json", "result.json", "test.json"):
            path = folder.parent / filename
            if path.is_file():
                report = json.loads(path.read_text())
                if report.get("run_id") == saved["run_id"]:
                    self.metadata[filename] = report
        self.metadata["seed"] = (
            self.metadata.get("run.json", {}).get("contract", {}).get("config", {}).get("seed", saved.get("seed"))
        )

    def info(self):
        performance = {}
        selection = self.metadata.get("result.json", {}).get("selection_metrics", {})
        test = self.metadata.get("test.json", {}).get("poem", {})
        if selection:
            performance["validation_poem_macro_f1"] = selection["macro_f1"]
        if test:
            performance.update(test_poem_macro_f1=test["macro_f1"], test_poems=test["observations"])
        return dict(
            model_type="cnn",
            model="Eight-class CNN",
            seed=self.metadata["seed"],
            labels=self.labels,
            parameters=sum(p.numel() for p in self.model.parameters()),
            max_tokens=self.vocabulary.max_tokens,
            temperature=1.0,
            calibrated=False,
            prediction_unit="poem",
            device=self.device,
            performance=performance,
            input_hint="Four full ordered pādas. For Seesamu: four body pādas, eight body half-lines, "
            "or eight body half-lines followed by four closing geeti lines.",
            aggregation_note="One prediction from all four ordered pādas. Seesamu closing geeti is excluded. "
            "Per-line predictions are unavailable for this poem-level CNN.",
            confidence_note="Uncalibrated class probabilities; a validated confidence interval is unavailable.",
            rejection_note="Restricted eight-class model; reliable other-metre rejection is not established.",
        )

    def predict(self, text):
        started = time.perf_counter()
        raw = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
        if len(raw) not in {4, 8, 12}:
            raise ValueError(
                "The eight-class CNN needs four full ordered pādas, or eight Seesamu body half-lines "
                "(optionally followed by four closing geeti lines). Select an original four-class model "
                "for a single pādam."
            )
        lines = raw if len(raw) == 4 else [" ".join(raw[i : i + 2]) for i in range(0, 8, 2)]
        ids, _ = collate([([self.vocabulary.encode(line) for line in lines], 0)])
        with self.lock, torch.inference_mode():
            logits = self.model(ids.to(self.device)).float().cpu().numpy()[0]
        shifted = logits.astype(np.float64) - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        winner = int(probabilities.argmax())
        ordered = np.sort(probabilities)
        unknown = sum(c not in self.vocabulary.tokens for line in lines for c in line)
        return dict(
            predicted_class=self.labels[winner],
            confidence=float(probabilities[winner]),
            confidence_interval=None,
            calibrated=False,
            classes=[dict(label=label, probability=float(probabilities[i])) for i, label in enumerate(self.labels)],
            mode="poem",
            line_count=4,
            input_line_count=len(raw),
            representation="four ordered full padas"
            if len(raw) == 4
            else "four body padas paired from eight half-lines",
            closing_geeti_lines_excluded=raw[8:] if len(raw) == 12 else [],
            line_scores_available=False,
            line_agreement=None,
            margin=float(ordered[-1] - ordered[-2]),
            entropy_bits=float(-(probabilities * np.log2(np.maximum(probabilities, 1e-12))).sum()),
            unknown_character_fraction=unknown / sum(map(len, lines)),
            temperature=1.0,
            seed=self.metadata["seed"],
            raw_poem_logits=logits.tolist(),
            lines=[
                dict(number=i + 1, text=line, predicted_class=None, probabilities=None, logits=None)
                for i, line in enumerate(lines)
            ],
            elapsed_ms=(time.perf_counter() - started) * 1000,
            confidence_note=self.info()["confidence_note"],
            rejection_note=self.info()["rejection_note"],
        )


class ExpandedModelCollection:
    """Route independent label spaces; ensemble only compatible original packages.

    Eight-class joint scores must never be averaged with four-class line scores,
    or presented as four independent line predictions.
    """

    def __init__(self, expanded, legacy, device="cpu", cpu_threads=4, strategy="single"):
        self.expanded = [ExpandedPredictor(folder, device, cpu_threads) for folder in expanded]
        self.legacy = ModelCollection(legacy, device, cpu_threads) if legacy else None
        self.members = self.expanded + (self.legacy.members if self.legacy else [])
        self.ids = [
            f"eight-class-seed-{p.metadata['seed']}" if p.metadata["seed"] is not None else f"eight-class-{i + 1}"
            for i, p in enumerate(self.expanded)
        ]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Expanded packages must have distinct seeds")
        if strategy == "ensemble" and (self.legacy is None or len(self.legacy.members) < 2):
            raise ValueError("Ensemble requires at least two compatible original four-class packages")
        self.default_strategy = "ensemble" if strategy == "ensemble" else self.ids[0]
        self.example = (
            self.legacy.example
            if self.legacy
            else "శ్రీ కైవల్య పదంబుఁ జేరుటకునై చింతించెదన్ లోక ర\n"
            "క్షైకారంభకు, భక్త పాలన కళా సంరంభకున్, దానవో\n"
            "ద్రేకస్తంభకుఁ, గేళి లోల విలసద్దృగ్జాల సంభూత నా\n"
            "నా కంజాత భవాండ కుంభకు, మహానందాంగనాడింభకున్."
        )

    def info(self):
        info = self.expanded[0].info()
        strategies = []
        for identifier, predictor in zip(self.ids, self.expanded):
            details = predictor.info()
            strategies.append(
                dict(
                    id=identifier,
                    name=f"Eight-class CNN · seed {predictor.metadata['seed']}",
                    performance=details["performance"],
                    labels=details["labels"],
                    calibrated=False,
                    prediction_unit="poem",
                    max_tokens=details["max_tokens"],
                    input_hint=details["input_hint"],
                    aggregation_note=details["aggregation_note"],
                )
            )
        if self.legacy:
            legacy_info = self.legacy.info()
            strategies.extend(
                dict(
                    s,
                    name=s["name"] + " · original 4 classes",
                    labels=legacy_info["labels"],
                    calibrated=True,
                    prediction_unit="line",
                    max_tokens=legacy_info["max_tokens"],
                )
                for s in legacy_info["strategies"]
            )
        info.update(
            strategies=strategies,
            default_strategy=self.default_strategy,
            loaded_models=len(self.members),
            loaded_seeds=[p.metadata.get("seed") for p in self.members],
            example=self.example,
            uncertainty_note="Compare seeds only within the same label space and architecture; "
            "the seed range is not a statistical confidence interval.",
        )
        return info

    def predict(self, text, strategy=None):
        strategy = strategy or self.default_strategy
        if strategy == "single":
            strategy = self.ids[0]
        if strategy not in self.ids:
            if self.legacy:
                return self.legacy.predict(text, strategy)
            raise ValueError("Requested prediction strategy is not loaded")
        result = self.expanded[self.ids.index(strategy)].predict(text)
        classes = [dict(c, seed_range=[c["probability"], c["probability"]]) for c in result["classes"]]
        result.update(
            classes=classes,
            strategy=strategy,
            strategy_name=next(s["name"] for s in self.info()["strategies"] if s["id"] == strategy),
            seed_score_range=[result["confidence"], result["confidence"]],
            compared_seeds=1,
            prediction_models=1,
            top_two_margin=result["margin"],
            seed_agreement=1.0,
            threshold_check=dict(threshold=None, score_above_saved_threshold=None, validated_rejection=False),
            seeds=[
                dict(
                    seed=result["seed"],
                    predicted_class=result["predicted_class"],
                    probabilities=[c["probability"] for c in classes],
                    temperature=1.0,
                    raw_line_logits=None,
                    raw_poem_logits=result["raw_poem_logits"],
                )
            ],
            model="Eight-class CNN",
            uncertainty_note=self.info()["uncertainty_note"],
        )
        return result


def load_collection(folders, device="cpu", cpu_threads=4, strategy="single"):
    expanded, legacy = [], []
    for folder in folders:
        config = json.loads((Path(folder) / "config.json").read_text())
        (expanded if config.get("architecture") == "ordered_poem_cnn_v1" else legacy).append(folder)
    if expanded:
        return ExpandedModelCollection(expanded, legacy, device, cpu_threads, strategy)
    return ModelCollection(folders, device, cpu_threads, strategy)
