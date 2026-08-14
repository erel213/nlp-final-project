from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.bilstm.dataset import DLPTextDataset, collate_fn
from models.bilstm.model import BiLSTMCharCNN
from models.bilstm.vocab import Vocabulary

LABEL_COLS = ["benign", "PII", "financial", "confidential"]
_BATCH_SIZE = 64


class BiLSTMDLPDetector:
    """Load a trained Bi-LSTM + char-CNN checkpoint and run inference.

    Args:
        checkpoint_path: path to best_model.pt saved by train.py
        vocab_path: path to vocab.json saved alongside the checkpoint
        device: "cpu" or "cuda"
    """

    LABEL_COLS = LABEL_COLS

    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.vocab = Vocabulary.load(vocab_path)

        # Reconstruct the exact architecture the checkpoint was trained with. The
        # char-CNN ablation variant has no char-CNN, so read run_meta.json if present
        # (falls back to the full-model default for legacy checkpoints).
        meta_path = Path(checkpoint_path).parent / "run_meta.json"
        use_char_cnn = True
        if meta_path.exists():
            use_char_cnn = json.loads(meta_path.read_text()).get("use_char_cnn", True)
        self.use_char_cnn = use_char_cnn

        self.model = BiLSTMCharCNN(
            self.vocab.n_words, self.vocab.n_chars, use_char_cnn=use_char_cnn
        )
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, emails: list[str]) -> np.ndarray:
        """Return sigmoid probabilities, shape (N, 4), columns in LABEL_COLS order."""
        dummy_labels = np.zeros((len(emails), 4), dtype=np.float32)
        ds = DLPTextDataset(emails, dummy_labels, self.vocab)
        loader = DataLoader(ds, batch_size=_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        parts: list[np.ndarray] = []
        with torch.no_grad():
            for word_ids, char_ids, mask, _ in loader:
                logits = self.model(
                    word_ids.to(self.device),
                    char_ids.to(self.device),
                    mask.to(self.device),
                )
                parts.append(torch.sigmoid(logits).cpu().numpy())

        return np.concatenate(parts, axis=0).astype(np.float32)
