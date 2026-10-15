"""Standalone Transformers loading for the ordered eight-class poem CNN.

Copyright 2026 Pranav Surampudi. SPDX-License-Identifier: LicenseRef-Chandassu-Citation
Depends only on PyTorch, Transformers and the standard library.
"""

import json
import shutil
import unicodedata
from pathlib import Path

import torch
from torch import nn
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

LABELS = ["ఉత్పలమాల", "చంపకమాల", "మత్తేభము", "శార్దూలం", "ఆటవెలది", "తేటగీతి", "కందము", "సీసము"]


class ChandassuPoemCNNConfig(PretrainedConfig):
    model_type = "chandassu_poem_cnn"

    def __init__(self, vocab_size=2, width=64, kernel_sizes=None, dropout=0.2, labels=None, max_tokens=512, **kwargs):
        labels = LABELS if labels is None else labels
        if labels != LABELS:
            raise ValueError("Requires the eight trained labels in their saved order")
        kwargs.setdefault("id2label", dict(enumerate(labels)))
        kwargs.setdefault("label2id", {label: i for i, label in enumerate(labels)})
        kwargs.setdefault("pad_token_id", 0)
        super().__init__(**kwargs)
        self.labels = labels
        self.vocab_size = vocab_size
        self.width = width
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        self.dropout = dropout
        self.max_tokens = max_tokens
        self.architecture = "ordered_poem_cnn_v1"
        self.calibrated = False
        self.prediction_unit = "poem"
        self.temperature = 1.0


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).replace("\ufeff", "").split())


class ChandassuPoemTokenizer(PreTrainedTokenizer):
    """Whole-poem tokenizer: tensors have shape [batch, 4 ordered padas, tokens].

    Pass one newline-separated poem string or a batch of complete poem strings.
    Four full padas, eight Seesamu body half-lines, and 8+4 Seesamu layouts are
    supported. In the latter case only the first eight body half-lines are used.
    """

    vocab_files_names = {"vocabulary_file": "vocabulary.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocabulary_file, **kwargs):
        self.vocabulary_file = str(vocabulary_file)
        saved = json.loads(Path(vocabulary_file).read_text())
        if (
            not saved
            or any(not isinstance(c, str) or len(c) != 1 for c in saved)
            or any(type(i) is not int for i in saved.values())
            or sorted(saved.values()) != list(range(2, len(saved) + 2))
        ):
            raise ValueError("Invalid character vocabulary")
        self.tokens = {"[PAD]": 0, "[UNK]": 1, **saved}
        self.reverse_tokens = {i: c for c, i in self.tokens.items()}
        self.limit = kwargs.get("model_max_length", 512)
        kwargs.setdefault("model_max_length", self.limit)
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("padding_side", "right")
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def get_vocab(self):
        return dict(self.tokens, **self.added_tokens_encoder)

    def tokenize(self, text, **kwargs):
        text = normalize(text)
        if not text or len(text) > self.limit:
            raise ValueError(f"Pada length outside 1..{self.limit}; no truncation")
        return list(text)

    def _tokenize(self, text):
        return self.tokenize(text)

    def _convert_token_to_id(self, token):
        return self.tokens.get(token, 1)

    def _convert_id_to_token(self, index):
        return self.reverse_tokens.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def __call__(
        self, text, text_pair=None, padding=True, truncation=False, max_length=None, return_tensors="pt", **kwargs
    ):
        if text_pair is not None:
            raise ValueError("Text pairs are unsupported; supply a complete poem")
        if truncation not in (None, False, "do_not_truncate"):
            raise ValueError("Truncation is unsupported; supply complete padas")
        if return_tensors not in (None, "pt"):
            raise ValueError("Use return_tensors='pt' or None")
        if kwargs.get("is_split_into_words") or kwargs.get("stride") or kwargs.get("return_overflowing_tokens"):
            raise ValueError("Split words, stride and overflowing chunks cannot represent a complete poem")
        if self.padding_side != "right" or kwargs.get("padding_side", "right") != "right":
            raise ValueError("The trained CNN requires right padding")
        poems = [text] if isinstance(text, str) else text
        if not isinstance(poems, (list, tuple)) or not poems or any(not isinstance(p, str) for p in poems):
            raise ValueError("Supply a poem string or a list of complete poem strings")
        encoded = []
        for poem in poems:
            lines = [normalize(t) for t in poem.splitlines() if normalize(t)]
            if len(lines) not in {4, 8, 12}:
                raise ValueError(
                    "Supply four full ordered padas, or eight Seesamu body half-lines "
                    "optionally followed by four closing geeti lines"
                )
            if len(lines) != 4:
                lines = [" ".join(lines[i : i + 2]) for i in range(0, 8, 2)]
            encoded.append([[self.tokens.get(c, 1) for c in self.tokenize(line)] for line in lines])
        longest = max(len(line) for poem in encoded for line in poem)
        if max_length is not None and (type(max_length) is not int or max_length < longest):
            raise ValueError("max_length cannot truncate a complete pada")
        if padding in {False, None, "do_not_pad"}:
            if return_tensors is not None and any(len(line) != longest for poem in encoded for line in poem):
                raise ValueError("Use padding=True for mixed-length padas in a tensor batch")
            length = None
        elif padding in {True, "longest", "max_length"}:
            length = (max_length or self.limit) if padding == "max_length" else longest
        else:
            raise ValueError("Unsupported padding mode")
        if length is not None:
            encoded = [[line + [0] * (length - len(line)) for line in poem] for poem in encoded]
        masks = [[[int(t != 0) for t in line] for line in poem] for poem in encoded]
        output = {"input_ids": encoded, "attention_mask": masks}
        if return_tensors == "pt":
            output = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return BatchEncoding(output)

    def encode(self, text, **kwargs):
        return self(text, **dict(kwargs, return_tensors=None))["input_ids"][0]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        path = Path(save_directory) / ((filename_prefix + "-" if filename_prefix else "") + "vocabulary.json")
        if path.resolve() != Path(self.vocabulary_file).resolve():
            shutil.copyfile(self.vocabulary_file, path)
        return (str(path),)


class ChandassuPoemCNNForSequenceClassification(PreTrainedModel):
    """One uncalibrated eight-class logit vector per complete poem."""

    config_class = ChandassuPoemCNNConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(config.width, config.width, k, padding=k // 2) for k in config.kernel_sizes]
        )
        features = 2 * config.width * len(config.kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(4 * features, 128), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(128, len(config.labels))
        )
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None, **kwargs):
        if input_ids.ndim != 3 or input_ids.shape[1] != 4:
            raise ValueError("Expected [batch, 4 ordered padas, tokens]; tokenize complete poems")
        valid = input_ids.ne(0)
        if not valid.any(2).all():
            raise ValueError("Empty pada")
        if attention_mask is not None and not torch.equal(attention_mask.bool(), valid):
            raise ValueError("Attention mask must match non-padding token IDs")
        batch = input_ids.shape[0]
        flat = input_ids.reshape(-1, input_ids.shape[-1])
        mask = flat.ne(0)
        x = self.dropout(self.embedding(flat).masked_fill(~mask.unsqueeze(-1), 0)).transpose(1, 2)
        features = []
        for conv in self.convolutions:
            h = torch.relu(conv(x))
            maximum = h.masked_fill(~mask.unsqueeze(1), float("-inf")).amax(2)
            mean = h.masked_fill(~mask.unsqueeze(1), 0).sum(2) / mask.sum(1, keepdim=True)
            features.extend([maximum, mean])
        logits = self.classifier(self.dropout(torch.cat(features, 1).reshape(batch, -1)))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        if return_dict is False or (return_dict is None and not self.config.use_return_dict):
            return (loss, logits) if loss is not None else (logits,)
        return SequenceClassifierOutput(loss=loss, logits=logits)
