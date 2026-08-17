"""Compute per-model probabilities for a split, fresh and in memory.

Every call runs each detector's ``predict_proba`` over the split — there is no
on-disk probability cache. This trades wall-clock (BERT/RoBERTa re-run inference
on every notebook execution) for correctness by construction: probabilities can
never be silently stale after a checkpoint retrain or a rule-based code change,
which was the failure mode of the removed ``ensemble/cache`` mechanism.

Row-alignment guarantee: all models score the same ordered ``source_text`` list
within a single call, so ``fuse`` (which silently assumes row alignment) is
always safe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.preprocessing import load_split, load_train_holdout
from evaluation import LABEL_COLS, from_dataframe

# The ensemble fits per-category weights + thresholds on a DEV partition that must
# be disjoint from the TEST reporting split. TEST is the untouched ai4privacy
# `validation` split; DEV is the seeded 10% held-out slice of `train` (the same
# slice neural checkpoint selection used). `train_holdout` is a synthetic split
# name routed to `load_train_holdout` rather than `load_dataset`.
_TRAIN_HOLDOUT_SPLIT = "train_holdout"
_TRAIN_HOLDOUT_FRAC = 0.10
_TRAIN_HOLDOUT_SEED = 42


def _load_split_df(split: str):
    """Load a DataFrame for a split name, routing the synthetic DEV split."""
    if split == _TRAIN_HOLDOUT_SPLIT:
        _, holdout_df = load_train_holdout(
            frac=_TRAIN_HOLDOUT_FRAC, seed=_TRAIN_HOLDOUT_SEED
        )
        return holdout_df
    return load_split(split)


REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAMES = ["bert", "roberta", "bilstm", "rule_based"]

_BERT_CKPT = REPO_ROOT / "models" / "bert" / "checkpoint" / "best_model.pt"
_ROBERTA_CKPT = REPO_ROOT / "models" / "roberta" / "checkpoint" / "best_model.pt"
_BILSTM_CKPT = REPO_ROOT / "models" / "bilstm" / "checkpoint" / "best_model.pt"
_BILSTM_VOCAB = REPO_ROOT / "models" / "bilstm" / "checkpoint" / "vocab.json"


def _build_detector(name: str, device: str):
    """Construct a detector by name (imports are lazy so a missing checkpoint
    for one model does not block the others)."""
    if name == "bert":
        from models.bert.predict import BertDLPDetector

        return BertDLPDetector(str(_BERT_CKPT), device=device)
    if name == "roberta":
        from models.roberta.predict import RobertaDLPDetector

        return RobertaDLPDetector(str(_ROBERTA_CKPT), device=device)
    if name == "bilstm":
        from models.bilstm.predict import BiLSTMDLPDetector

        return BiLSTMDLPDetector(str(_BILSTM_CKPT), str(_BILSTM_VOCAB), device=device)
    if name == "rule_based":
        from models.rule_based.detector import RuleBasedDetector

        return RuleBasedDetector()
    raise ValueError(f"Unknown model name: {name!r}")


def compute_probas(
    split: str = "validation",
    device: str = "cpu",
    models: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Run every detector on ``split`` and return ``(model_probas, y_true, texts)``.

    ``model_probas`` maps each model name to an ``(N, 4)`` float32 array in
    ``LABEL_COLS`` order, row-aligned with ``y_true`` and ``texts``. ``models``
    selects a subset (default: all four).
    """
    df = _load_split_df(split)
    texts = df["source_text"].tolist()
    y_true = from_dataframe(df).astype(int)

    names = list(models) if models else list(MODEL_NAMES)
    unknown = [m for m in names if m not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {MODEL_NAMES}.")

    model_probas: dict[str, np.ndarray] = {}
    for name in names:
        print(f"[probas] running {name} on {len(texts):,} texts (split={split!r}) ...")
        detector = _build_detector(name, device)
        probas = detector.predict_proba(texts).astype(np.float32)
        if probas.shape != (len(texts), len(LABEL_COLS)):
            raise ValueError(
                f"{name} predict_proba returned {probas.shape}, "
                f"expected {(len(texts), len(LABEL_COLS))}"
            )
        model_probas[name] = probas

    return model_probas, y_true, texts
