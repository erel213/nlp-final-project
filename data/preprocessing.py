from __future__ import annotations

import json
from pathlib import Path

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
