from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from datasets import load_dataset

from data.entity_mapping import CONFIDENTIAL_ENTITY_TYPES, FINANCIAL_ENTITY_TYPES

DATA_RAW = Path(__file__).parent / "ai4privacy" / "raw"

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]


def parse_mask(mask) -> list[dict]:
    """Return entity list from privacy_mask regardless of HuggingFace storage format."""
    if mask is None or isinstance(mask, float):
        return []
    if isinstance(mask, np.ndarray):
        return list(mask)
    if isinstance(mask, list):
        return mask
    if isinstance(mask, str):
        try:
            return json.loads(mask)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def build_labels(privacy_mask) -> dict[str, int]:
    """Map a privacy_mask value to the project's 5-label multi-label schema."""
    entities = parse_mask(privacy_mask)
    entity_types = {e["label"] for e in entities}
    has_any = bool(entities)
    return {
        "benign":       int(not has_any),
        "PII":          int(has_any),
        "financial":    int(bool(entity_types & FINANCIAL_ENTITY_TYPES)),
        "confidential": int(bool(entity_types & CONFIDENTIAL_ENTITY_TYPES)),
    }


def load_split(split: str, language: str = "English") -> pd.DataFrame:
    """Load a dataset split, filter by language, and attach multi-labels.

    Returns a DataFrame with original columns plus label columns:
        benign, PII, financial, confidential  (int 0/1 per row)
    """
    ds = load_dataset("ai4privacy/pii-masking-300k", cache_dir=str(DATA_RAW))
    df = ds[split].filter(lambda x: x["language"] == language).to_pandas()
    label_df = df["privacy_mask"].apply(build_labels).apply(pd.Series)
    return pd.concat([df, label_df], axis=1).reset_index(drop=True)


def load_train_holdout(
    frac: float = 0.10,
    seed: int = 42,
    language: str = "English",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a seeded held-out slice from ``train`` for model selection.

    The ai4privacy dataset exposes only ``train`` and ``validation``. To keep the
    ``validation`` split an untouched TEST set (touched exactly once, for final
    reporting of all models and the ensemble), neural early-stopping / checkpoint
    selection must use a slice of ``train`` instead — per ``.claude/rules/model-bert.md``
    ("use a held-out 10% of training data").

    Returns ``(train_fit_df, selection_holdout_df)``:
        - ``train_fit_df`` — the (1-frac) fraction used to fit model parameters.
        - ``selection_holdout_df`` — the ``frac`` fraction used ONLY for early
          stopping / checkpoint selection. Never reported on.

    The split is a deterministic seeded permutation, so every training script and
    the ensemble weight-fitting DEV cache see the same partition.
    """
    df = load_split("train", language=language)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_holdout = max(1, int(round(frac * len(df))))
    holdout_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]
    train_fit_df = df.iloc[fit_idx].reset_index(drop=True)
    selection_holdout_df = df.iloc[holdout_idx].reset_index(drop=True)
    return train_fit_df, selection_holdout_df


def _stratify_key(df: pd.DataFrame, k: int) -> np.ndarray:
    """Collapse the 4-bit label vector into a single stratification label.

    Each row maps to a string like ``"1_0_1_0"`` (benign_PII_financial_confidential).
    ``StratifiedKFold`` requires every class to have at least ``k`` members, so any
    label-combo rarer than ``k`` is merged into a shared ``"rare"`` stratum. This keeps
    the imbalanced financial/confidential rows spread evenly across folds without adding
    a multilabel-stratification dependency.
    """
    combos = df[LABEL_COLS].astype(int).astype(str).agg("_".join, axis=1)
    counts = combos.value_counts()
    rare = set(counts[counts < k].index)
    return combos.map(lambda c: "rare" if c in rare else c).to_numpy()


def load_kfold_splits(
    k: int = 5,
    seed: int = 42,
    inner_dev_frac: float = 0.10,
    language: str = "English",
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield ``k`` stratified cross-validation folds over the ``train`` split.

    Cross-validation is a *supplementary* robustness experiment: it runs entirely on
    the dataset-provided ``train`` split so the ``validation`` split stays the untouched
    TEST set for final single-number reporting (see ``load_train_holdout``). Labels are
    stratified on the collapsed label-combo key so rare financial/confidential rows are
    spread evenly across folds.

    Within each fold there are THREE disjoint partitions, mirroring the DEV/TEST
    discipline of ``load_train_holdout`` but repeated per fold:

        - ``test_fold_df``   — the held-out fold; the ONLY partition metrics are reported on.
        - ``inner_dev_df``   — a seeded ``inner_dev_frac`` slice of the other ``k-1`` folds,
          used for BOTH neural checkpoint selection AND ensemble weight/threshold fitting.
        - ``train_fit_df``   — the remainder of the other folds; neural parameter training.

    Fitting fusion weights on ``inner_dev_df`` (never on ``test_fold_df``) keeps
    weight-fitting disjoint from evaluation, so the ensemble's per-fold score is not
    optimistically biased.

    Yields ``(fold_idx, train_fit_df, inner_dev_df, test_fold_df)`` for
    ``fold_idx`` in ``0..k-1``. Each DataFrame is reset-indexed with ``LABEL_COLS``
    attached — the same schema every other loader returns.
    """
    from sklearn.model_selection import StratifiedKFold

    df = load_split("train", language=language)
    strat = _stratify_key(df, k)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    for fold_idx, (pool_idx, test_idx) in enumerate(skf.split(np.zeros(len(df)), strat)):
        test_fold_df = df.iloc[test_idx].reset_index(drop=True)

        # Carve a seeded inner-dev slice from the pooled (k-1) training folds. A
        # per-fold generator keeps the carve deterministic yet distinct across folds.
        perm = rng.permutation(len(pool_idx))
        n_dev = max(1, int(round(inner_dev_frac * len(pool_idx))))
        dev_idx = pool_idx[perm[:n_dev]]
        fit_idx = pool_idx[perm[n_dev:]]

        inner_dev_df = df.iloc[dev_idx].reset_index(drop=True)
        train_fit_df = df.iloc[fit_idx].reset_index(drop=True)
        yield fold_idx, train_fit_df, inner_dev_df, test_fold_df
