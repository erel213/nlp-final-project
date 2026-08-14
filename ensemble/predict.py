"""Confidence-weighted late-fusion ensemble detector.

Loads all four trained detectors plus the learned per-category weights and fuses
their per-label probabilities via ``evaluation.fuse`` (single source of truth for
the fusion formula). Mirrors the ``predict_proba`` interface of the per-model
detectors so callers can treat the ensemble like any other model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation import LABEL_COLS, fuse, load_weights

REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_DIR = Path(__file__).resolve().parent

_BERT_CKPT = REPO_ROOT / "models" / "bert" / "checkpoint" / "best_model.pt"
_ROBERTA_CKPT = REPO_ROOT / "models" / "roberta" / "checkpoint" / "best_model.pt"
_BILSTM_CKPT = REPO_ROOT / "models" / "bilstm" / "checkpoint" / "best_model.pt"
_BILSTM_VOCAB = REPO_ROOT / "models" / "bilstm" / "checkpoint" / "vocab.json"


class EnsembleDLPDetector:
    """Weighted late-fusion ensemble of BERT, RoBERTa, Bi-LSTM, and rule-based.

    Args:
        weights_path: JSON of ``{model: {label: float}}`` (from train_weights).
        thresholds_path: optional JSON of ``{label: float}`` per-label thresholds;
            falls back to 0.5 for every label if absent.
        device: "cpu" or "cuda" for the neural detectors.
    """

    LABEL_COLS = LABEL_COLS

    def __init__(
        self,
        weights_path: str | Path = _MODULE_DIR / "weights.json",
        thresholds_path: str | Path = _MODULE_DIR / "thresholds.json",
        device: str = "cpu",
    ) -> None:
        from models.bert.predict import BertDLPDetector
        from models.bilstm.predict import BiLSTMDLPDetector
        from models.roberta.predict import RobertaDLPDetector
        from models.rule_based.detector import RuleBasedDetector

        self.weights = load_weights(weights_path)
        self.detectors = {
            "bert": BertDLPDetector(str(_BERT_CKPT), device=device),
            "roberta": RobertaDLPDetector(str(_ROBERTA_CKPT), device=device),
            "bilstm": BiLSTMDLPDetector(str(_BILSTM_CKPT), str(_BILSTM_VOCAB), device=device),
            "rule_based": RuleBasedDetector(),
        }

        thresholds_path = Path(thresholds_path)
        if thresholds_path.exists():
            self.thresholds = json.loads(thresholds_path.read_text())
        else:
            self.thresholds = {label: 0.5 for label in LABEL_COLS}

    def predict_proba(self, emails: list[str]) -> np.ndarray:
        """Return fused per-label probabilities, shape (N, 4), LABEL_COLS order."""
        model_probas = {name: det.predict_proba(emails) for name, det in self.detectors.items()}
        _, fused = fuse(model_probas, self.weights, threshold=0.5)
        return fused.astype(np.float32)

    def predict(self, emails: list[str]) -> np.ndarray:
        """Return binary predictions using per-label thresholds, shape (N, 4)."""
        fused = self.predict_proba(emails)
        cols = [
            (fused[:, i] >= self.thresholds.get(label, 0.5)).astype(int)
            for i, label in enumerate(LABEL_COLS)
        ]
        return np.stack(cols, axis=1)
