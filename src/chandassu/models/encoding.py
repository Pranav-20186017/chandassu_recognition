"""Train-only character encoding and whole-poem collation."""

from dataclasses import dataclass, field

import torch
from torch.nn.utils.rnn import pad_sequence

from .. import LABEL_TO_ID
from ..data import normalize_text
from .direct import ByteVocabulary


@dataclass
class CharacterVocabulary:
    max_tokens: int = 256
    tokens: dict = field(default_factory=dict)

    def fit(self, texts):
        self.tokens = {t: i + 2 for i, t in enumerate(sorted({c for s in texts for c in normalize_text(s)}))}
        if not self.tokens:
            raise ValueError("Empty character vocabulary")
        return self

    def encode(self, text):
        text = normalize_text(text)
        if not text or len(text) > self.max_tokens:
            raise ValueError(f"Line length {len(text)} outside 1..{self.max_tokens}; no truncation")
        return [self.tokens.get(c, 1) for c in text]


def vocabulary_for(model_type, max_tokens, fitting=(), saved=None):
    if model_type == "byt5":
        return ByteVocabulary(max_tokens)
    vocabulary = CharacterVocabulary(max_tokens)
    if saved is not None:
        if not isinstance(saved, dict) or not saved or any(not isinstance(c, str) or len(c) != 1 for c in saved):
            raise ValueError("Character vocabulary must contain single-codepoint tokens")
        if any(type(i) is not int for i in saved.values()) or sorted(saved.values()) != list(range(2, len(saved) + 2)):
            raise ValueError("Character token IDs must be unique and contiguous from 2")
        vocabulary.tokens = saved
    else:
        vocabulary.fit(t for poem in fitting for t in poem.lines)
    return vocabulary


def collate_poems(examples):
    sequences, labels = [], []
    for lines, label in examples:
        if len(lines) != 4:
            raise ValueError("Each training example must be a complete four-line poem")
        sequences.extend(lines)
        labels.extend([label] * 4)
    return pad_sequence([torch.tensor(s, dtype=torch.long) for s in sequences], batch_first=True), torch.tensor(labels)


def encoded_examples(poems, vocabulary):
    return [
        ([vocabulary.encode(t) for t in p.lines], LABEL_TO_ID[p.label]) for p in sorted(poems, key=lambda p: p.poem_id)
    ]
