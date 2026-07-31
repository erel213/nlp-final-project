from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from models.bilstm.vocab import Vocabulary

MAX_WORD_LEN = 25
MAX_SEQ_LEN = 512


class DLPTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: Vocabulary) -> None:
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self._word_ids: list[torch.Tensor] = []
        self._char_ids: list[torch.Tensor] = []

        for text in texts:
            tokens = (text or " ").lower().split()[:MAX_SEQ_LEN]
            if not tokens:
                tokens = [" "]
            self._word_ids.append(
                torch.tensor([vocab.encode_word(t) for t in tokens], dtype=torch.long)
            )
            self._char_ids.append(
                torch.tensor(
                    [vocab.encode_chars(t, MAX_WORD_LEN) for t in tokens], dtype=torch.long
                )
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self._word_ids[idx], self._char_ids[idx], self.labels[idx]


def collate_fn(batch):
    word_ids_list, char_ids_list, labels_list = zip(*batch)

    max_len = max(w.size(0) for w in word_ids_list)
    B = len(word_ids_list)

    word_ids_padded = torch.zeros(B, max_len, dtype=torch.long)
    char_ids_padded = torch.zeros(B, max_len, MAX_WORD_LEN, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (w, c) in enumerate(zip(word_ids_list, char_ids_list)):
        seq_len = w.size(0)
        word_ids_padded[i, :seq_len] = w
        char_ids_padded[i, :seq_len] = c
        mask[i, :seq_len] = True

    return word_ids_padded, char_ids_padded, mask, torch.stack(labels_list)
