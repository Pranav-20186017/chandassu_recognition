"""Single-seed selection and equal calibrated-probability ensembles."""

import json
import time
from pathlib import Path

import numpy as np

from .. import LABELS
from .predictor import Predictor


class ModelCollection:
    def __init__(self, folders, device="cpu", cpu_threads=4, strategy="single"):
        if not folders:
            raise ValueError("At least one model package is required")
        self.members = [Predictor(folder, device, cpu_threads) for folder in folders]
        families = {member.config["model_type"] for member in self.members}
        if len(families) != 1:
            raise ValueError("Ensemble members must use the same model family")
        self.ids = [
            f"seed-{p.metadata['seed']}" if p.metadata.get("seed") is not None else f"model-{i + 1}"
            for i, p in enumerate(self.members)
        ]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Each loaded ensemble member must have a distinct seed")
        if strategy not in {"single", "ensemble"} or strategy == "ensemble" and len(self.members) < 2:
            raise ValueError("Ensemble strategy requires at least two distinct model packages")
        self.default_strategy = "ensemble" if strategy == "ensemble" else self.ids[0]
        self.example = ""
        example_path = Path(__file__).parents[3] / "data/v1/train.jsonl"
        if example_path.exists():
            with example_path.open() as handle:
                self.example = "\n".join(json.loads(next(handle))["lines"])

    def info(self):
        info = self.members[0].info()
        strategies = [
            {
                "id": identifier,
                "name": f"Single model · seed {p.metadata.get('seed', 'unspecified')}",
                "performance": p.metadata.get("performance", {}),
            }
            for identifier, p in zip(self.ids, self.members)
        ]
        if len(self.members) > 1:
            performance = {}
            published = self.members[0].metadata.get("published_ensemble", {})
            actual = {str(p.metadata.get("seed")): p.metadata.get("source_checkpoint_sha256") for p in self.members}
            if actual == published.get("checkpoint_hashes"):
                performance = published["performance"]
            strategies.append(
                {"id": "ensemble", "name": f"Ensemble · {len(self.members)} seeds", "performance": performance}
            )
        info.update(
            strategies=strategies,
            default_strategy=self.default_strategy,
            loaded_models=len(self.members),
            loaded_seeds=[p.metadata.get("seed") for p in self.members],
            example=self.example,
            model=f"Final {info['model_type'].upper()}",
            max_tokens=min(p.vocabulary.max_tokens for p in self.members),
            uncertainty_note="The seed range shows variation between trained models, not a statistical confidence interval.",
        )
        return info

    def predict(self, text, strategy=None):
        started = time.perf_counter()
        strategy = strategy or self.default_strategy
        if strategy == "single":
            strategy = self.ids[0]
        if strategy not in self.ids and not (strategy == "ensemble" and len(self.members) > 1):
            raise ValueError("Requested prediction strategy is not loaded")
        # All loaded seeds provide the score-stability comparison in either mode.
        outputs = [member.predict(text) for member in self.members]
        active = outputs if strategy == "ensemble" else [outputs[self.ids.index(strategy)]]
        line_scores = np.mean([[line["probabilities"] for line in output["lines"]] for output in active], axis=0)
        scores = line_scores.mean(0)
        seed_scores = np.array([[c["probability"] for c in output["classes"]] for output in outputs])
        winner = int(scores.argmax())
        per_class = [
            {
                "label": label,
                "probability": float(scores[i]),
                "seed_range": [float(seed_scores[:, i].min()), float(seed_scores[:, i].max())],
            }
            for i, label in enumerate(LABELS)
        ]
        result = dict(active[0])
        recipe = (
            self.members[self.ids.index(strategy)].metadata.get("original_recipe", {}) if strategy != "ensemble" else {}
        )
        threshold = recipe.get("threshold", {}).get("threshold")
        unknown = [d["unknown_character_fraction"] for d in active if d["unknown_character_fraction"] is not None]
        ordered = np.sort(scores)
        info = self.info()
        result.update(
            predicted_class=LABELS[winner],
            confidence=float(scores[winner]),
            classes=per_class,
            strategy=strategy,
            strategy_name=next(s["name"] for s in info["strategies"] if s["id"] == strategy),
            seed_score_range=per_class[winner]["seed_range"],
            compared_seeds=len(outputs),
            prediction_models=len(active),
            top_two_margin=float(ordered[-1] - ordered[-2]),
            margin=float(ordered[-1] - ordered[-2]),
            entropy_bits=float(-(scores * np.log2(np.maximum(scores, 1e-12))).sum()),
            line_agreement=float((line_scores.argmax(1) == winner).mean()),
            seed_agreement=float((seed_scores.argmax(1) == winner).mean()),
            unknown_character_fraction=float(np.mean(unknown)) if unknown else None,
            temperature=None if strategy == "ensemble" else active[0]["temperature"],
            seed=None if strategy == "ensemble" else active[0]["seed"],
            threshold_check={
                "threshold": threshold,
                "score_above_saved_threshold": None if threshold is None else bool(scores[winner] >= threshold),
                "validated_rejection": False,
            },
            lines=[
                dict(
                    line,
                    predicted_class=LABELS[int(line_scores[i].argmax())],
                    probabilities=line_scores[i].tolist(),
                    logits=None if strategy == "ensemble" else line["logits"],
                    raw_logits=None if strategy == "ensemble" else line["logits"],
                )
                for i, line in enumerate(active[0]["lines"])
            ],
            seeds=[
                {
                    "seed": output["seed"],
                    "predicted_class": output["predicted_class"],
                    "probabilities": [c["probability"] for c in output["classes"]],
                    "temperature": output["temperature"],
                    "raw_line_logits": [line["logits"] for line in output["lines"]],
                }
                for output in outputs
            ],
            elapsed_ms=(time.perf_counter() - started) * 1000,
            model=info["model"],
            uncertainty_note=info["uncertainty_note"],
        )
        return result
