from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.bert.dataset import DLPDataset, collate_fn, MODEL_NAME
from models.bert.model import BertForDLPClassification

LABEL_COLS = ["benign", "PII", "financial", "confidential"]
_BATCH_SIZE = 32


class BertDLPDetector:
    """Load a fine-tuned BERT checkpoint and run inference.

    Args:
        checkpoint_path: path to best_model.pt saved by train.py
        device: "cpu" or "cuda"
    """

    LABEL_COLS = LABEL_COLS

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = BertForDLPClassification()
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, emails: list[str]) -> np.ndarray:
        """Return sigmoid probabilities, shape (N, 4), columns in LABEL_COLS order."""
        dummy_labels = np.zeros((len(emails), 4), dtype=np.float32)
        ds = DLPDataset(emails, dummy_labels, tokenizer=self.tokenizer)
        loader = DataLoader(ds, batch_size=_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        parts: list[np.ndarray] = []
        with torch.no_grad():
            for input_ids, attention_mask, _ in loader:
                logits = self.model(
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                )
                parts.append(torch.sigmoid(logits).cpu().numpy())

        return np.concatenate(parts, axis=0).astype(np.float32)
