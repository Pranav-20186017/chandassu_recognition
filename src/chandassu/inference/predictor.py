"""Calibrated line and poem inference through the same artifact loader."""

import threading
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .. import LABELS
from ..artifacts import load_package
from ..data import normalize_text
from ..evaluation.metrics import softmax


class Predictor:
    def __init__(self, folder, device="cpu", cpu_threads=4):
        torch.set_num_threads(cpu_threads)
        self.model, self.vocabulary, self.config, self.metadata = load_package(folder, device)
        self.device = device
        self.lock = threading.Lock()

    def logits(self, lines, batch_size=32):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        sequences = [torch.tensor(self.vocabulary.encode(t), dtype=torch.long) for t in lines]
        outputs = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                x = pad_sequence(sequences[start : start + batch_size], batch_first=True).to(self.device)
                outputs.append(self.model(x).float().cpu().numpy())
        if not outputs:
            raise ValueError("No nonempty lines")
        return np.concatenate(outputs)

    def info(self):
        return {
            "model_type": self.config["model_type"],
            "seed": self.metadata.get("seed"),
            "labels": list(LABELS),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "max_tokens": self.vocabulary.max_tokens,
            "temperature": self.config["temperature"],
            "device": self.device,
            "confidence_note": "Calibrated probabilities; no statistical confidence interval is available.",
            "rejection_note": "Restricted four-class model; cannot reliably reject other metres.",
        }

    def predict(self, text):
        started = time.perf_counter()
        lines = [normalize_text(t) for t in text.splitlines() if normalize_text(t)]
        if len(lines) not in {1, 4}:
            raise ValueError("Paste one line or exactly four nonempty lines")
        logits = self.logits(lines, batch_size=4)
        probabilities = softmax(logits, self.config["temperature"])
        mean = probabilities.mean(0)
        winner = int(mean.argmax())
        ordered = np.sort(mean)
        unknown = (
            sum(sum(c not in self.vocabulary.tokens for c in t) for t in lines)
            if self.config["model_type"] == "cnn"
            else None
        )
        return {
            "predicted_class": LABELS[winner],
            "confidence": float(mean[winner]),
            "confidence_interval": None,
            "classes": [{"label": label, "probability": float(mean[i])} for i, label in enumerate(LABELS)],
            "mode": "poem" if len(lines) == 4 else "line",
            "line_count": len(lines),
            "line_agreement": int((probabilities.argmax(1) == winner).sum()),
            "margin": float(ordered[-1] - ordered[-2]),
            "entropy_bits": float(-(mean * np.log2(np.maximum(mean, 1e-12))).sum()),
            "unknown_character_fraction": unknown / sum(map(len, lines)) if unknown is not None else None,
            "temperature": self.config["temperature"],
            "seed": self.metadata.get("seed"),
            "lines": [
                {
                    "number": i + 1,
                    "text": line,
                    "predicted_class": LABELS[int(probabilities[i].argmax())],
                    "probabilities": probabilities[i].tolist(),
                    "logits": logits[i].tolist(),
                }
                for i, line in enumerate(lines)
            ],
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "confidence_note": self.info()["confidence_note"],
            "rejection_note": self.info()["rejection_note"],
        }
