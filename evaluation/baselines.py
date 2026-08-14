"""Trivial reference baselines for the SentinelMail sensitive-sentence task.

These exist to quantify how much of the near-ceiling neural F1 is reachable
WITHOUT any learning — i.e. how much headroom the fine-tuned models actually
add over dumb floors. They answer reviewer comment 004 ("labels are a
deterministic function of the entity mask, so scores look trivially high").

Two baselines are provided:

  * ``MajorityBaseline`` — a per-label constant predictor using the TRAINING-set
    prior. With ai4privacy English base rates (benign 11.6%, PII 88.4%,
    financial 19.0%, confidential 44.3%) the majority label per column is
    ``[benign=0, PII=1, financial=0, confidential=0]`` — predicted for every row.
    This is the uninformed floor.

  * ``RegexTextBaseline`` — a hard rule classifier that reads ONLY ``source_text``.
    It MUST NOT look at ``privacy_mask``: the ground-truth labels are defined as a
    deterministic function of ``privacy_mask`` entity types, so a mask-reading
    classifier would trivially score ~1.0 by construction (an oracle, not a
    baseline). Reading surface text only measures the pattern-matching floor.

    This is DELIBERATELY dumber than, and distinct from, the project's Presidio
    ``rule_based`` model (``models/rule_based/detector.py``), which uses a
    statistical NER pipeline (PERSON/LOCATION recognizers etc.). This baseline is
    pure hand-written patterns with NO name/place detector — the harshest floor,
    so its PII recall collapses on name-driven sentences and the transformer PII
    headroom is exposed.

Both baselines follow the standard detector contract used across the repo:
``predict_batch(texts) -> list[dict]`` with the 4-key schema and ``y_proba=None``,
so they flow through ``from_dicts`` + ``compute_all_metrics`` unchanged.
"""

from __future__ import annotations

import re

import numpy as np

LABEL_COLS: list[str] = ["benign", "PII", "financial", "confidential"]


# ---------------------------------------------------------------------------
# 1. Majority-class (prior) baseline
# ---------------------------------------------------------------------------
class MajorityBaseline:
    """Per-label constant predictor: for each label, always predict the
    training-set majority class (1 if the training prior >= 0.5 else 0)."""

    def __init__(self, priors: dict[str, float]):
        self.priors = priors
        # Per-label majority class from the TRAINING prior.
        self.constant = {
            label: int(priors[label] >= 0.5) for label in LABEL_COLS
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, int | None]]:
        row = {**self.constant, "y_proba": None}
        return [dict(row) for _ in texts]


# ---------------------------------------------------------------------------
# 2. Text-only regex baseline (NO access to privacy_mask)
# ---------------------------------------------------------------------------
# Financial: card-number / SSN-shaped digit patterns + card-issuer keyword lexicon.
_CARD_NUMBER_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN_RE = re.compile(r"\b\d{3}[ -]?\d{2}[ -]?\d{4}\b")
_CARD_ISSUER_LEXICON = [
    "visa", "mastercard", "master card", "amex", "american express",
    "discover", "diners club", "jcb", "maestro", "unionpay", "credit card",
    "debit card", "card issuer", "cardholder",
]

# Confidential: keyword triggers + document-number shapes (passport / license / id).
_CONFIDENTIAL_LEXICON = [
    "passport", "driver license", "driver's license", "driving license",
    "driving licence", "driver licence", "id card", "identity card",
    "national id", "password", "passcode", "pass code", "pin code",
]
_DOCNUM_RE = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")  # e.g. passport / license number shapes

# PII surface signals (pattern-only — NO name/place detector on purpose).
_EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\.\(\) ]{6,}\d)\b")
_URL_RE = re.compile(r"\bhttps?://|\bwww\.")


def _contains_any(text_lower: str, lexicon: list[str]) -> bool:
    return any(term in text_lower for term in lexicon)


class RegexTextBaseline:
    """Hard-coded pattern classifier over ``source_text`` only.

    Intentionally a "pure-patterns" variant: it has NO TitleCase name-guess
    heuristic (see note in comment 004). That makes it the harshest floor for
    PII, which is dominated by name-driven sentences that carry no regexable
    surface pattern.
    """

    # NOTE: privacy_mask is deliberately never referenced here. The ground-truth
    # labels are derived from privacy_mask, so reading it would score 1.0 by
    # construction and measure nothing.
    def predict_one(self, text: str) -> dict[str, int | None]:
        low = text.lower()

        financial = int(
            bool(_CARD_NUMBER_RE.search(text))
            or bool(_SSN_RE.search(text))
            or _contains_any(low, _CARD_ISSUER_LEXICON)
        )
        confidential = int(
            _contains_any(low, _CONFIDENTIAL_LEXICON)
            or bool(_DOCNUM_RE.search(text))
        )
        pii_surface = (
            bool(_EMAIL_RE.search(text))
            or bool(_PHONE_RE.search(text))
            or bool(_URL_RE.search(text))
        )
        # PII fires on surface PII patterns OR any financial/confidential hit.
        # No name/place detector on purpose -> collapses on name-driven sentences.
        PII = int(pii_surface or financial or confidential)
        benign = int(not (PII or financial or confidential))

        return {
            "benign": benign,
            "PII": PII,
            "financial": financial,
            "confidential": confidential,
            "y_proba": None,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, int | None]]:
        return [self.predict_one(t) for t in texts]


def training_priors_english() -> dict[str, float]:
    """Return English label base rates from the processed label table.

    Used to parameterise ``MajorityBaseline``. Falls back to the documented
    figures if the parquet is unavailable.
    """
    from pathlib import Path

    import pandas as pd

    path = (
        Path(__file__).resolve().parent.parent
        / "data" / "ai4privacy" / "processed" / "labels_en.parquet"
    )
    if path.exists():
        df = pd.read_parquet(path)
        return {label: float(df[label].mean()) for label in LABEL_COLS}
    return {"benign": 0.1164, "PII": 0.8836, "financial": 0.1902, "confidential": 0.4431}
