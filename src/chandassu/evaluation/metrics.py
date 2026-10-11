"""Four fixed classes; poem probabilities average four calibrated line scores."""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .. import LABELS


def softmax(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    if z.ndim != 2 or z.shape[1] != 4 or not np.isfinite(z).all():
        raise ValueError("Expected finite four-class logits")
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def classification(targets, probabilities):
    predictions = probabilities.argmax(1)
    return {
        "observations": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=range(4), average="macro", zero_division=0)),
        "cross_entropy": float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean()),
        "per_class": classification_report(
            targets, predictions, labels=range(4), target_names=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=range(4)).tolist(),
        "class_order": list(LABELS),
    }


def line_poem_metrics(logits, targets, temperature=1.0):
    if not len(targets) or len(targets) % 4:
        raise ValueError("Metrics require complete poems")
    y = np.asarray(targets, dtype=int)
    grouped = y.reshape(-1, 4)
    if not np.all(grouped == grouped[:, :1]):
        raise ValueError("Four-line target mismatch")
    p = softmax(logits, temperature)
    return {"line": classification(y, p), "poem": classification(grouped[:, 0], p.reshape(-1, 4, 4).mean(1))}


def fit_temperature(logits, targets):
    targets = np.asarray(targets).reshape(-1, 4)
    if not np.all(targets == targets[:, :1]):
        raise ValueError("Calibration requires complete poems")

    def loss(log_temperature):
        p = softmax(logits, np.exp(log_temperature)).reshape(-1, 4, 4).mean(1)
        return float(-np.log(np.maximum(p[np.arange(len(targets)), targets[:, 0]], 1e-12)).mean())

    result = minimize_scalar(loss, bounds=(-3, 3), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    return {
        "temperature": float(np.exp(result.x)),
        "before_nll": loss(0),
        "after_nll": float(result.fun),
        "poems": len(targets),
        "resource": "Separate calibration works; excluded from optimization and selection",
    }


def better_checkpoint(row, best, f1_delta, loss_delta):
    if best is None:
        return True
    improvement = row["validation_poem_f1"] - best["validation_poem_f1"]
    return improvement > f1_delta or (
        abs(improvement) <= f1_delta and row["validation_eval_loss"] < best["validation_eval_loss"] - loss_delta
    )
