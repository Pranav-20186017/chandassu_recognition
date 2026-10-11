"""Standalone Transformers classes exported with Chandassu model packages.

Copyright 2026 Pranav Surampudi. SPDX-License-Identifier: LicenseRef-Chandassu-Citation
This module intentionally depends only on PyTorch, Transformers and the standard library.
"""

import json
import shutil
import unicodedata
from pathlib import Path

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, T5Config, T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ChandassuConfig(PretrainedConfig):
    """Keep the original fitting configuration and calibrated temperature."""

    def __init__(self, labels=None, temperature=1.0, calibration_temperature=None, **kwargs):
        labels = labels or ["ఉత్పలమాల", "చంపకమాల", "మత్తేభము", "శార్దూలం"]
        kwargs.setdefault("id2label", dict(enumerate(labels)))
        kwargs.setdefault("label2id", {label: i for i, label in enumerate(labels)})
        super().__init__(**kwargs)
        self.labels = labels
        self.calibration_temperature = temperature if calibration_temperature is None else calibration_temperature
        if len(labels) != 4 or not 0 < self.calibration_temperature < float("inf"):
            raise ValueError("Require four labels and a finite positive temperature")


class ChandassuCNNConfig(ChandassuConfig):
    model_type = "cnn"


class ChandassuByT5Config(ChandassuConfig):
    model_type = "byt5"


class ChandassuTokenizer(PreTrainedTokenizer):
    """Exact training-time NFC, whitespace, character and UTF-8 byte encoding."""

    vocab_files_names = {"vocabulary_file": "vocabulary.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocabulary_file, **kwargs):
        self.vocabulary_file = str(vocabulary_file)
        saved = json.loads(Path(vocabulary_file).read_text())
        self.kind = saved["kind"]
        self.limit = saved["max_tokens"]
        if self.kind == "character":
            self.tokens = {"[PAD]": 0, "[UNK]": 1, **saved["tokens"]}
            defaults = {"pad_token": "[PAD]", "unk_token": "[UNK]"}
        elif self.kind == "utf8_bytes":
            self.tokens = {"<pad>": 0, "</s>": 1, "<unk>": 2}
            self.tokens.update({f"<byte:{i}>": i + 3 for i in range(256)})
            defaults = {"pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>"}
        else:
            raise ValueError("Unsupported vocabulary kind")
        self.reverse_tokens = {value: key for key, value in self.tokens.items()}
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        kwargs.setdefault("model_max_length", self.limit)
        kwargs.setdefault("padding_side", "right")
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def get_vocab(self):
        return dict(self.tokens, **self.added_tokens_encoder)

    def tokenize(self, text, **kwargs):
        # Override special-token splitting: literal text must have the same IDs as training.
        text = " ".join(unicodedata.normalize("NFC", text).replace("\ufeff", "").split())
        if self.kind == "character":
            if not text or len(text) > self.limit:
                raise ValueError(f"Line length outside 1..{self.limit}; no truncation")
            return list(text)
        encoded = text.encode("utf-8")
        if len(encoded) + 1 > self.limit:
            raise ValueError("Byte line exceeds capacity; no truncation")
        return [f"<byte:{value}>" for value in encoded]

    def _tokenize(self, text):
        return self.tokenize(text)

    def _convert_token_to_id(self, token):
        return self.tokens.get(token, self.tokens[str(self.unk_token)])

    def _convert_id_to_token(self, index):
        return self.reverse_tokens.get(index, str(self.unk_token))

    def convert_tokens_to_string(self, tokens):
        if self.kind == "character":
            return "".join(tokens)
        return bytes(int(t[6:-1]) for t in tokens if t.startswith("<byte:")).decode("utf-8", errors="replace")

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            raise ValueError("Encode poem lines separately; text pairs are unsupported")
        return token_ids_0 + ([1] if self.kind == "utf8_bytes" else [])

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, True)
        return [0] * len(token_ids_0) + ([1] if self.kind == "utf8_bytes" else [])

    def prepare_for_model(self, *args, **kwargs):
        if kwargs.get("truncation") not in (None, False, "do_not_truncate"):
            raise ValueError("Truncation is unsupported; supply a complete line within the input limit")
        return super().prepare_for_model(*args, **kwargs)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        name = (filename_prefix + "-" if filename_prefix else "") + "vocabulary.json"
        destination = Path(save_directory) / name
        if destination.resolve() != Path(self.vocabulary_file).resolve():
            shutil.copyfile(self.vocabulary_file, destination)
        return (str(destination),)


class ChandassuClassifier(PreTrainedModel):
    """Four raw line logits; use config.calibration_temperature for calibrated probabilities."""

    config_class = ChandassuConfig
    _tied_weights_keys = ["encoder.encoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        if config.model_type == "cnn":
            self.embedding = nn.Embedding(config.vocab_size, config.width, padding_idx=0)
            self.convolutions = nn.ModuleList(
                [nn.Conv1d(config.width, config.width, k, padding=k // 2) for k in config.kernel_sizes]
            )
            self.classifier = nn.Linear(config.width * len(config.kernel_sizes), 4)
        elif config.model_type == "byt5":
            self.encoder = T5EncoderModel(T5Config.from_dict(config.encoder_config))
            self.classifier = nn.Linear(self.encoder.config.d_model, 4)
        else:
            raise ValueError("Unsupported model family")
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None, **kwargs):
        valid = input_ids.ne(0)
        if not valid.any(1).all():
            raise ValueError("Empty line")
        if attention_mask is not None and not torch.equal(attention_mask.bool(), valid):
            raise ValueError("Attention mask must match non-padding token IDs")
        if self.config.model_type == "cnn":
            embedded = self.embedding(input_ids).masked_fill(~valid.unsqueeze(-1), 0.0)
            features = self.dropout(embedded).transpose(1, 2)
            pooled = torch.cat(
                [
                    torch.relu(conv(features)).masked_fill(~valid.unsqueeze(1), float("-inf")).amax(2)
                    for conv in self.convolutions
                ],
                dim=1,
            )
        else:
            hidden = self.encoder(input_ids=input_ids, attention_mask=valid.long(), return_dict=True).last_hidden_state
            pooled = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        logits = self.classifier(self.dropout(pooled))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        if return_dict is False or (return_dict is None and not self.config.use_return_dict):
            return (loss, logits) if loss is not None else (logits,)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class ChandassuCNNForSequenceClassification(ChandassuClassifier):
    config_class = ChandassuCNNConfig


class ChandassuByT5ForSequenceClassification(ChandassuClassifier):
    config_class = ChandassuByT5Config
