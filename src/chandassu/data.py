"""Complete poems, conservative normalization, and explicit work-group splits."""

import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from . import LABELS
from .io import sha256


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).replace("\ufeff", "").split())


def layout_key(text):
    return "".join(c for c in normalize_text(text) if unicodedata.category(c)[0] in {"L", "M"})


@dataclass(frozen=True)
class Poem:
    poem_id: str
    source: str
    author: str
    lines: tuple[str, ...]
    label: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record):
        required = {"poem_id", "source", "author", "lines", "label"}
        if not required <= record.keys():
            raise ValueError(f"Poem requires fields: {sorted(required)}")
        if any(not isinstance(record[k], str) or not record[k].strip() for k in required - {"lines"}):
            raise ValueError("Poem ID, source, author and label must be nonempty strings")
        lines = record["lines"]
        if (
            not isinstance(lines, list)
            or len(lines) != 4
            or any(not isinstance(t, str) or not normalize_text(t) for t in lines)
        ):
            raise ValueError(f"{record['poem_id']}: exactly four nonempty ordered lines required")
        if record["label"] not in LABELS:
            raise ValueError(f"Unsupported target label: {record['label']}")
        return cls(
            record["poem_id"],
            record["source"],
            record["author"],
            tuple(normalize_text(t) for t in lines),
            record["label"],
            {k: v for k, v in record.items() if k not in required},
        )


def read_poems(path):
    poems = []
    with Path(path).open(encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if line.strip():
                try:
                    poems.append(Poem.from_dict(json.loads(line)))
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{n}: {error}") from error
    if not poems or len({p.poem_id for p in poems}) != len(poems):
        raise ValueError(f"{path}: empty corpus or duplicate poem IDs")
    return sorted(poems, key=lambda p: p.poem_id)


def fingerprint(poems):
    # Match the original H100 line fingerprint exactly, including ordering/serialization.
    rows = [
        {"poem_id": p.poem_id, "source": p.source, "author": p.author, "line_no": i + 1, "text": t, "label": p.label}
        for p in sorted(poems, key=lambda p: p.poem_id)
        for i, t in enumerate(p.lines)
    ]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def summary(poems):
    counts = Counter(p.label for p in poems)
    return {
        "poems": len(poems),
        "lines": len(poems) * 4,
        "works": len({p.source for p in poems}),
        "authors": len({p.author for p in poems}),
        "classes": {label: counts[label] for label in LABELS},
    }


def assert_isolated(parts, require_all_classes=True):
    seen_ids, seen_sources, seen_lines, seen_editions = {}, {}, {}, {}
    for role, poems in parts.items():
        if require_all_classes and {p.label for p in poems} != set(LABELS):
            raise ValueError(f"{role} must contain all four classes")
        for poem in poems:
            edition = poem.metadata.get("edition_family") or poem.source
            for key, seen, kind in (
                (poem.poem_id, seen_ids, "poem"),
                (poem.source, seen_sources, "work"),
                (edition, seen_editions, "edition family"),
            ):
                if key in seen and (seen[key] != role or kind == "poem"):
                    raise ValueError(f"{kind} leakage/duplicate across {seen[key]} and {role}: {key}")
                seen[key] = role
            for line in poem.lines:
                key = layout_key(line)
                if key in seen_lines and seen_lines[key] != role:
                    raise ValueError(f"Line/layout duplicate across {seen_lines[key]} and {role}")
                seen_lines[key] = role


def work_groups(poems):
    parent = {p.source: p.source for p in poems}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    seen_lines, seen_editions = {}, {}
    for poem in poems:
        edition = poem.metadata.get("edition_family") or poem.source
        if edition in seen_editions:
            union(poem.source, seen_editions[edition])
        seen_editions[edition] = poem.source
        for line in poem.lines:
            key = layout_key(line)
            if key in seen_lines:
                union(poem.source, seen_lines[key])
            seen_lines[key] = poem.source
    return [find(p.source) for p in poems]


def holdout(poems, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be between zero and one")
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    fit, held = next(splitter.split(np.zeros(len(poems)), groups=work_groups(poems)))
    parts = {"fit": [poems[i] for i in fit], "held_out": [poems[i] for i in held]}
    assert_isolated(parts)
    return parts["fit"], parts["held_out"]


def plain_folds(poems, count, seed):
    splitter = GroupKFold(n_splits=count, shuffle=True, random_state=seed)
    result = []
    for fitting, validation in splitter.split(np.zeros(len(poems)), groups=work_groups(poems)):
        parts = {"fit": [poems[i] for i in fitting], "selection": [poems[i] for i in validation]}
        assert_isolated(parts)
        result.append(parts)
    return result


class Corpus:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        if self.manifest["schema_version"] != 2 or self.manifest["labels"] != list(LABELS):
            raise ValueError("Unsupported prepared corpus")
        for name, expected in self.manifest["files"].items():
            path = self.folder / name
            if path.resolve().parent != self.folder.resolve() and self.folder.resolve() not in path.resolve().parents:
                raise ValueError("Corpus manifest path escapes data directory")
            if sha256(path) != expected:
                raise ValueError(f"Dataset checksum mismatch: {name}")

    def poems(self, role):
        if role not in {"train", "validation", "test"}:
            raise ValueError("Unknown data role")
        return read_poems(self.folder / f"{role}.jsonl")

    def final_parts(self, split_manifest, calibration_fraction, split_seed):
        train, validation = self.poems("train"), self.poems("validation")
        assert_isolated({"train": train, "validation": validation})
        if split_manifest:
            partition = json.loads((self.folder / split_manifest).read_text())
            lookup = {p.poem_id: p for p in train + validation}
            parts = {role: [lookup[k] for k in ids] for role, ids in partition["ids"].items()}
            ids = [p.poem_id for part in parts.values() for p in part]
            if len(ids) != len(set(ids)) or set(ids) != set(lookup):
                raise ValueError("Frozen partitions do not exactly allocate development poems")
            if {p.poem_id for p in parts["selection"]} != {p.poem_id for p in validation}:
                raise ValueError("Frozen selection does not equal validation split")
        elif calibration_fraction:
            fitting, calibration = holdout(train, calibration_fraction, split_seed)
            parts = {"fit": fitting, "selection": validation, "calibration": calibration}
        else:
            parts = {"fit": train, "selection": validation, "calibration": []}
        assert_isolated({k: v for k, v in parts.items() if v})
        return parts


def poem_dict(poem):
    value = asdict(poem)
    value.update(value.pop("metadata"))
    return value
