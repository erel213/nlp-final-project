from __future__ import annotations

import re
from typing import Callable, TypedDict

import numpy as np
import pandas as pd

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]

_NEGATION_RE = re.compile(
    r"\b(not|no|never|don'?t|doesn'?t|isn'?t|wasn'?t|haven'?t|won'?t|cannot|can'?t)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(
    r"\b(if|suppose|supposing|imagine|imagining|hypothetically|assume|assuming|"
    r"example|e\.g\.|for instance|let'?s say|pretend|what if)\b",
    re.IGNORECASE,
)
_OBFUSCATED_NUM_RE = re.compile(r"\d[\s\-\.]{1,2}\d[\s\-\.]{1,2}\d")
_OBFUSCATED_SSN_RE = re.compile(
    r"s[\s\-_]?s[\s\-_]?n|p[\s\-_]?i[\s\-_]?i", re.IGNORECASE
)
_IMPLICIT_KEYWORDS = re.compile(
    r"\b(salary|wage|payroll|wire transfer|routing number|account number|"
    r"bank account|iban|swift|credential|passphrase|api key|secret key|"
    r"access token|private key|social security|national id|taxpayer id)\b",
    re.IGNORECASE,
)


class ErrorRecord(TypedDict):
    idx: int
    text_snippet: str
    label: str
    error_type: str
    y_true: int
    y_pred: int
    explain: list[dict] | None


def _classify_fp(text: str) -> str:
    if _NEGATION_RE.search(text):
        return "FP_negation"
    if _HYPOTHETICAL_RE.search(text):
        return "FP_hypothetical"
    return "other_FP"


def _classify_fn(text: str) -> str:
    if _OBFUSCATED_NUM_RE.search(text) or _OBFUSCATED_SSN_RE.search(text):
        return "FN_obfuscated"
    if _IMPLICIT_KEYWORDS.search(text):
        return "FN_implicit"
    return "other_FN"


def classify_errors(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    explain_fn: Callable[[str], list[dict]] | None = None,
    max_examples: int | None = None,
) -> dict[str, list[ErrorRecord]]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    buckets: dict[str, list[ErrorRecord]] = {
        "FP_negation": [],
        "FP_hypothetical": [],
        "FN_obfuscated": [],
        "FN_implicit": [],
        "other_FP": [],
        "other_FN": [],
    }
    counts: dict[str, dict[str, int]] = {k: {} for k in buckets}

    for idx, text in enumerate(texts):
        for col_i, label in enumerate(LABEL_COLS):
            true_val = int(y_true[idx, col_i])
            pred_val = int(y_pred[idx, col_i])

            if true_val == pred_val:
                continue

            if max_examples is not None:
                bucket_key = (
                    _classify_fp(text) if pred_val > true_val else _classify_fn(text)
                )
                current = counts[bucket_key].get(label, 0)
                if current >= max_examples:
                    continue
                counts[bucket_key][label] = current + 1

            if pred_val > true_val:
                error_type = _classify_fp(text)
                spans = explain_fn(text) if explain_fn is not None else None
            else:
                error_type = _classify_fn(text)
                spans = None

            record: ErrorRecord = {
                "idx": idx,
                "text_snippet": text[:120],
                "label": label,
                "error_type": error_type,
                "y_true": true_val,
                "y_pred": pred_val,
                "explain": spans,
            }
            buckets[error_type].append(record)

    return buckets


def error_summary(error_results: dict[str, list[ErrorRecord]]) -> pd.DataFrame:
    rows = []
    for error_type, records in error_results.items():
        label_counts: dict[str, int] = {}
        for r in records:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        for label, count in label_counts.items():
            rows.append({"error_type": error_type, "label": label, "count": count})
    if not rows:
        return pd.DataFrame(columns=["error_type", "label", "count"])
    df = pd.DataFrame(rows).sort_values(["error_type", "label"]).reset_index(drop=True)
    return df
