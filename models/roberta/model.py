from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

N_LABELS = 4
HIDDEN_SIZE = 768
MODEL_NAME = "roberta-base"

# Freeze embedding layer + encoder blocks 0-9; train blocks 10-11 + head
_FROZEN_BLOCK_COUNT = 10


class RobertaForDLPClassification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self._freeze_layers()
        self.classifier = nn.Linear(HIDDEN_SIZE, N_LABELS)

    def _freeze_layers(self) -> None:
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for i in range(_FROZEN_BLOCK_COUNT):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # <s> token, (B, 768)
        return self.classifier(cls_repr)               # (B, 4) logits
