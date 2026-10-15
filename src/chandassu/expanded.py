"""Opt-in eight-class, ordered whole-poem CNN; legacy four-class packages stay reproducible."""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from . import LABELS as VRUTTA_LABELS
from .data import assert_isolated, layout_key, normalize_text
from .io import sha256
from .models.encoding import CharacterVocabulary

LABELS = VRUTTA_LABELS + ("ఆటవెలది", "తేటగీతి", "కందము", "సీసము")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class ExpandedPoem:
    poem_id: str
    source: str
    author: str
    lines: tuple[str, ...]
    label: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row):
        required = {"poem_id", "source", "author", "lines", "label"}
        if not required <= row.keys():
            raise ValueError("Missing whole-poem fields")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in required - {"lines"}):
            raise ValueError("Identity, source, author and label must be nonempty strings")
        lines = row["lines"]
        if (
            not isinstance(lines, list)
            or len(lines) != 4
            or any(not isinstance(t, str) or not normalize_text(t) for t in lines)
        ):
            raise ValueError("Exactly four complete ordered padas required")
        if row["label"] not in LABELS:
            raise ValueError("Unsupported label")
        if "closing_geeti_lines" in row and (
            row["label"] != "సీసము"
            or not isinstance(row["closing_geeti_lines"], list)
            or len(row["closing_geeti_lines"]) != 4
            or any(not isinstance(t, str) or not layout_key(t) for t in row["closing_geeti_lines"])
        ):
            raise ValueError("Seesamu provenance requires four nonempty closure lines")
        return cls(
            *(row[k] for k in ["poem_id", "source", "author"]),
            tuple(map(normalize_text, lines)),
            row["label"],
            {k: v for k, v in row.items() if k not in required},
        )


def counts(poems):
    return dict(
        poems=len(poems),
        lines=sum(len(p.lines) for p in poems),
        works=len({p.source for p in poems}),
        classes={lab: sum(p.label == lab for p in poems) for lab in LABELS},
    )


def assert_parts(parts):
    assert_isolated(parts, require_all_classes=False)
    ids, line_roles = set(), {}
    for role, poems in parts.items():
        if role in {"fit", "selection", "test"} and {p.label for p in poems} != set(LABELS):
            raise ValueError(f"{role} needs all eight labels")
        for p in poems:
            if p.poem_id in ids:
                raise ValueError("Duplicate poem ID")
            ids.add(p.poem_id)
            for line in p.lines + tuple(p.metadata.get("closing_geeti_lines", [])):
                key = layout_key(line)
                if key in line_roles and line_roles[key] != role:
                    raise ValueError("Body/closure text crosses partitions")
                line_roles[key] = role


def load_corpus(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema_version"] != 3 or manifest["labels"] != list(LABELS):
        raise ValueError("Requires eight-class schema v3")
    for name, digest in manifest["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError(f"Data integrity failure: {name}")
    parts = {
        role: [ExpandedPoem.from_dict(json.loads(line)) for line in (folder / f"{role}.jsonl").open()]
        for role in ["fit", "selection", "calibration", "test"]
    }
    assert_parts(parts)
    if manifest["counts"] != {role: counts(rows) for role, rows in parts.items()}:
        raise ValueError("Data manifest counts differ")
    return parts, manifest


class OrderedPoemCNN(nn.Module):
    """Shared character convolutions, masked max/mean pooling, ordered four-pada head.

    A single loss supervises each complete poem. No label markers, source IDs,
    closure text, rule scores or engineered prosody features enter the model.
    """

    def __init__(self, vocabulary_size, width=64, kernel_sizes=(3, 5, 7), dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.convolutions = nn.ModuleList([nn.Conv1d(width, width, k, padding=k // 2) for k in kernel_sizes])
        features = 2 * width * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(4 * features, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, len(LABELS))
        )

    def forward(self, ids):
        if ids.ndim != 3 or ids.shape[1] != 4:
            raise ValueError("Expected batch of four ordered padas")
        batch = ids.shape[0]
        flat = ids.reshape(-1, ids.shape[-1])
        mask = flat.ne(0)
        if not mask.any(1).all():
            raise ValueError("Empty pada")
        x = self.dropout(self.embedding(flat).masked_fill(~mask.unsqueeze(-1), 0)).transpose(1, 2)
        features = []
        for conv in self.convolutions:
            h = torch.relu(conv(x))
            maximum = h.masked_fill(~mask.unsqueeze(1), float("-inf")).amax(2)
            mean = h.masked_fill(~mask.unsqueeze(1), 0).sum(2) / mask.sum(1, keepdim=True)
            features.extend([maximum, mean])
        return self.classifier(self.dropout(torch.cat(features, 1).reshape(batch, -1)))


def encode(poems, vocabulary):
    return [([vocabulary.encode(line) for line in poem.lines], LABEL_TO_ID[poem.label]) for poem in poems]


def collate(examples):
    capacity = max(len(line) for lines, _ in examples for line in lines)
    ids = torch.zeros(len(examples), 4, capacity, dtype=torch.long)
    labels = []
    for i, (lines, label) in enumerate(examples):
        if len(lines) != 4:
            raise ValueError("Incomplete poem")
        for j, line in enumerate(lines):
            ids[i, j, : len(line)] = torch.tensor(line)
        labels.append(label)
    return ids, torch.tensor(labels)


def class_weights(poems):
    c = Counter(p.label for p in poems)
    if set(c) != set(LABELS):
        raise ValueError("Fit requires all eight labels")
    # Square-root inverse frequency limits gradient domination by scarce classes.
    weights = torch.tensor([(len(poems) / (len(LABELS) * c[label])) ** 0.5 for label in LABELS])
    return weights / weights.mean()


def load_model(folder):
    from safetensors.torch import load_file

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if not (folder / name).resolve().is_relative_to(folder.resolve()) or sha256(folder / name) != expected:
            raise ValueError("Expanded model checksum mismatch")
    config = json.loads((folder / "config.json").read_text())
    if config["architecture"] != "ordered_poem_cnn_v1" or config["labels"] != list(LABELS):
        raise ValueError("Unsupported expanded CNN package")
    vocabulary = CharacterVocabulary(config["max_tokens"], json.loads((folder / "vocabulary.json").read_text()))
    model = OrderedPoemCNN(len(vocabulary.tokens) + 2, config["width"], config["kernel_sizes"], config["dropout"])
    model.load_state_dict(load_file(str(folder / "model.safetensors")))
    return model.eval(), vocabulary, config
