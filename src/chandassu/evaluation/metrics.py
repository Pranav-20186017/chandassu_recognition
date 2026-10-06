"""Four-class metrics and probability-averaged poem predictions."""
from collections import defaultdict
from .. import LABELS, LABEL_TO_ID


def classification_metrics(targets, predictions):
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Metrics need equally sized, nonempty targets and predictions")
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for true, pred in zip(targets, predictions):
        if true not in range(len(LABELS)) or pred not in range(len(LABELS)):
            raise ValueError("Class IDs must be in 0–3")
        matrix[true][pred] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {"accuracy": sum(matrix[i][i] for i in range(len(LABELS))) / len(targets),
            "macro_f1": sum(v["f1"] for v in per_class.values()) / len(LABELS),
            "per_class": per_class, "confusion_matrix": matrix,
            "confusion_matrix_axes": {"rows": "true", "columns": "predicted", "labels": LABELS}}


def reports(rows, probabilities):
    if len(rows) != len(probabilities):
        raise ValueError("One probability vector required per line")
    groups = defaultdict(list)
    line_predictions = []
    for row, probs in zip(rows, probabilities):
        if len(probs) != len(LABELS):
            raise ValueError("Expected four probabilities per line")
        pred = max(range(len(LABELS)), key=lambda i: probs[i])
        groups[row.poem_id].append((row, probs))
        line_predictions.append({"poem_id": row.poem_id, "line_no": row.line_no,
                                 "true_label": row.label, "predicted_label": LABELS[pred],
                                 "probabilities": dict(zip(LABELS, probs))})
    poem_predictions = []
    for pid, group in groups.items():
        if len(group) != 4 or len({r.label for r, _ in group}) != 1:
            raise ValueError("Poem aggregation requires four lines with the same label")
        means = [sum(p[i] for _, p in group) / 4 for i in range(len(LABELS))]
        pred = max(range(len(LABELS)), key=lambda i: means[i])
        poem_predictions.append({"poem_id": pid, "true_label": group[0][0].label,
                                 "predicted_label": LABELS[pred], "probabilities": dict(zip(LABELS, means))})
    metrics = {}
    for name, predictions in (("line", line_predictions), ("poem", poem_predictions)):
        metrics[name] = classification_metrics([LABEL_TO_ID[r["true_label"]] for r in predictions],
                                               [LABEL_TO_ID[r["predicted_label"]] for r in predictions])
    return metrics, {"lines": line_predictions, "poems": poem_predictions}
