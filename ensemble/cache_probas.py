"""Cache per-model validation probabilities to disk.

BERT/RoBERTa CPU inference over the full validation split is the bottleneck for
ensemble weight fitting, which needs to iterate over the same probabilities many
times. This module runs each detector's ``predict_proba`` once, saves the result
as ``.npy``, and provides an aligned loader.

Alignment guarantee: every model's row i must correspond to the same email. We
store an sha1 of the ordered ``source_text`` list in ``meta.json`` and re-check it
on load, so ``fuse`` (which silently assumes row alignment) is always safe.
"""

from __future__ import annotations

import hashlib
import json
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
CACHE_DIR = Path(__file__).resolve().parent / "cache"

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


def _texts_sha1(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")  # delimiter so order/boundaries matter
    return h.hexdigest()


def _paths(split: str) -> tuple[dict[str, Path], Path, Path]:
    proba_paths = {m: CACHE_DIR / f"probas_{m}_{split}.npy" for m in MODEL_NAMES}
    y_true_path = CACHE_DIR / f"y_true_{split}.npy"
    meta_path = CACHE_DIR / f"meta_{split}.json"
    return proba_paths, y_true_path, meta_path


def build_cache(split: str = "validation", device: str = "cpu", force: bool = False) -> None:
    """Run every detector on ``split`` and cache probabilities + labels.

    Idempotent: skips work if a valid cache already exists for the same emails,
    unless ``force=True``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    proba_paths, y_true_path, meta_path = _paths(split)

    df = _load_split_df(split)
    texts = df["source_text"].tolist()
    sha1 = _texts_sha1(texts)

    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        have_all = all(p.exists() for p in proba_paths.values()) and y_true_path.exists()
        if have_all and meta.get("source_text_sha1") == sha1:
            print(f"[cache] up to date for split={split!r} (n={len(texts)}); use force=True to rebuild.")
            return

    y_true = from_dataframe(df).astype(int)
    np.save(y_true_path, y_true)

    for name in MODEL_NAMES:
        print(f"[cache] running {name} on {len(texts)} emails ...")
        detector = _build_detector(name, device)
        probas = detector.predict_proba(texts).astype(np.float32)
        if probas.shape != (len(texts), len(LABEL_COLS)):
            raise ValueError(
                f"{name} predict_proba returned {probas.shape}, expected {(len(texts), len(LABEL_COLS))}"
            )
        np.save(proba_paths[name], probas)

    meta = {
        "split": split,
        "n": len(texts),
        "label_cols": LABEL_COLS,
        "models": MODEL_NAMES,
        "source_text_sha1": sha1,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[cache] wrote {len(MODEL_NAMES)} models + labels for split={split!r} to {CACHE_DIR}")


def load_probas(split: str = "validation") -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Load cached probabilities, labels, and texts, asserting row alignment.

    Returns ``(model_probas, y_true, texts)`` where ``model_probas`` maps each
    model name to an ``(N, 4)`` float array in ``LABEL_COLS`` order.
    """
    proba_paths, y_true_path, meta_path = _paths(split)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No cache for split={split!r} at {CACHE_DIR}. Run build_cache({split!r}) first."
        )

    meta = json.loads(meta_path.read_text())
    df = _load_split_df(split)
    texts = df["source_text"].tolist()

    if _texts_sha1(texts) != meta.get("source_text_sha1"):
        raise RuntimeError(
            f"Cache for split={split!r} is stale (source_text changed). "
            f"Rebuild with build_cache({split!r}, force=True)."
        )
    if meta.get("label_cols") != LABEL_COLS:
        raise RuntimeError(f"Cache label order {meta.get('label_cols')} != {LABEL_COLS}.")

    model_probas = {m: np.load(proba_paths[m]) for m in meta["models"]}
    y_true = np.load(y_true_path)

    n = len(texts)
    for name, arr in model_probas.items():
        if arr.shape != (n, len(LABEL_COLS)):
            raise RuntimeError(f"{name} cache shape {arr.shape} != {(n, len(LABEL_COLS))}")
    if y_true.shape != (n, len(LABEL_COLS)):
        raise RuntimeError(f"y_true cache shape {y_true.shape} != {(n, len(LABEL_COLS))}")

    return model_probas, y_true, texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the ensemble probability cache.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_cache(args.split, device=args.device, force=args.force)
