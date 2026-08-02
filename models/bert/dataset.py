from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MAX_SEQ_LEN = 512
MODEL_NAME = "bert-base-uncased"


class DLPDataset(Dataset):
    """Tokenizes texts with bert-base-uncased and pairs them with multi-label targets."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer | None = None,
    ) -> None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Tokenize without padding — collate_fn handles batch padding
        encoding = tokenizer(
            texts,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding=False,
        )
        self.input_ids: list[list[int]] = encoding["input_ids"]
        self.attention_masks: list[list[int]] = encoding["attention_mask"]
        self.labels = labels  # (N, 4) float32 numpy array

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.attention_masks[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def collate_fn(batch):
    input_ids_list, masks_list, labels_list = zip(*batch)

    max_len = max(t.size(0) for t in input_ids_list)
    B = len(input_ids_list)

    input_ids_padded = torch.zeros(B, max_len, dtype=torch.long)
    masks_padded = torch.zeros(B, max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, masks_list)):
        seq_len = ids.size(0)
        input_ids_padded[i, :seq_len] = ids
        masks_padded[i, :seq_len] = mask

    return input_ids_padded, masks_padded, torch.stack(labels_list)
