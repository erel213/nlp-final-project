from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

N_LABELS = 4
HIDDEN_SIZE = 768
MODEL_NAME = "bert-base-uncased"

# Freeze embedding layer + encoder blocks 0-9; train blocks 10-11 + head (ADR-001)
_FROZEN_BLOCK_COUNT = 10


class BertForDLPClassification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self._freeze_layers()
        self.classifier = nn.Linear(HIDDEN_SIZE, N_LABELS)

    def _freeze_layers(self) -> None:
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(_FROZEN_BLOCK_COUNT):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token, (B, 768)
        return self.classifier(cls_repr)               # (B, 4) logits
