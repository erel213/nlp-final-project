"""SentinelMail ensemble: confidence-weighted per-category late fusion.

The fusion math (``fuse``, ``run_ablation``, ``load_weights``) lives in the
``evaluation`` package and is reused here. This package adds the missing layer:
computing per-model probabilities, learning per-category weights, and the
``EnsembleDLPDetector`` runtime.
"""

from __future__ import annotations

from .probas import compute_probas
from .predict import EnsembleDLPDetector
from .train_weights import (
    fit_label_weights,
    fit_weights,
    fit_weights_lbfgsb,
    save_thresholds,
    save_weights,
    tune_thresholds,
)

__all__ = [
    "compute_probas",
    "EnsembleDLPDetector",
    "fit_label_weights",
    "fit_weights",
    "fit_weights_lbfgsb",
    "tune_thresholds",
    "save_weights",
    "save_thresholds",
]
