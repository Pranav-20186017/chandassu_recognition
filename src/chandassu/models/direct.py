"""Direct raw-text classifiers. Rule outputs are not model inputs."""

import torch
from torch import nn
from ..data import normalize_text


class TeluguMetreCNN(nn.Module):
    def __init__(self, vocabulary_size, width=64, kernel_sizes=(3, 5, 7), dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(width, width, k, padding=k // 2) for k in kernel_sizes]
        )
        self.classifier = nn.Linear(width * len(kernel_sizes), 4)

    def forward(self, ids):
        valid = ids.ne(0)
        if not valid.any(1).all():
            raise ValueError("Empty line")
        embedded = self.embedding(ids).masked_fill(~valid.unsqueeze(-1), 0.0)
        features = self.dropout(embedded).transpose(1, 2)
        pooled = torch.cat(
            [
                torch.relu(conv(features))
                .masked_fill(~valid.unsqueeze(1), float("-inf"))
                .amax(2)
                for conv in self.convolutions
            ],
            dim=1,
        )
        return self.classifier(self.dropout(pooled))


class ByteVocabulary:
    def __init__(self, max_tokens=1024):
        self.max_tokens = max_tokens

    def encode(self, text):
        ids = [b + 3 for b in normalize_text(text).encode("utf-8")] + [1]
        if len(ids) > self.max_tokens:
            raise ValueError("Byte line exceeds capacity; never silently truncate")
        return ids


class ByteMetreEncoder(nn.Module):
    def __init__(self, encoder, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.d_model, 4)

    def forward(self, ids):
        mask = ids.ne(0)
        if not mask.any(1).all():
            raise ValueError("Empty line")
        encoded = self.encoder(
            input_ids=ids, attention_mask=mask.long(), return_dict=True
        ).last_hidden_state
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.classifier(self.dropout(pooled))
