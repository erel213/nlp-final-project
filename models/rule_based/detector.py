from __future__ import annotations

from typing import Any

import numpy as np
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from data.entity_mapping import (
    PRESIDIO_CONFIDENTIAL_ENTITIES as CONFIDENTIAL_ENTITIES,
    PRESIDIO_FINANCIAL_ENTITIES as FINANCIAL_ENTITIES,
    PRESIDIO_PII_ENTITIES as PII_ENTITIES,
)

_EMPTY_LABELS: dict[str, int] = {
    "benign": 1,
    "PII": 0,
    "financial": 0,
    "health": 0,
    "confidential": 0,
}


def _build_analyzer(spacy_model: str) -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


class RuleBasedDetector:
    """Zero-training DLP baseline using Microsoft Presidio (regex + spaCy NER).

    Wraps Presidio's AnalyzerEngine and maps detected entity types to the
    project's 4-label schema: {benign, PII, financial, confidential}.
    """

    LABEL_COLS = ["benign", "PII", "financial", "confidential"]

    def __init__(self, score_threshold: float = 0.4, spacy_model: str = "en_core_web_lg") -> None:
        self.score_threshold = score_threshold
        self._analyzer = _build_analyzer(spacy_model)

    def predict(self, text: str) -> dict[str, int]:
        """Classify a single text. Returns label dict matching datasets.md schema."""
        results = self._analyzer.analyze(text=text, language="en", score_threshold=self.score_threshold)

        detected: set[str] = {r.entity_type for r in results}

        has_pii = bool(detected & PII_ENTITIES)
        has_fin = bool(detected & FINANCIAL_ENTITIES)
        has_conf = bool(detected & CONFIDENTIAL_ENTITIES)

        return {
            "benign":       int(not has_pii),
            "PII":          int(has_pii),
            "financial":    int(has_fin),
            "confidential": int(has_conf),
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, int]]:
        """Classify a list of texts."""
        return [self.predict(t) for t in texts]

    def predict_proba(self, emails: list[str]) -> np.ndarray:
        """Return binary 0.0/1.0 predictions, shape (N, 4), columns in LABEL_COLS order.

        These are hard flags, not calibrated probabilities (see ADR-003 in
        model-rule-based.md). For single-model evaluation, pass this array as
        y_pred and y_proba=None to compute_all_metrics. For the ensemble, it is
        consumed by evaluation.fuse as a 0/1 probability contribution.
        """
        preds = self.predict_batch(emails)
        return np.array(
            [[float(p[label]) for label in self.LABEL_COLS] for p in preds],
            dtype=np.float32,
        )

    def explain(self, text: str) -> list[dict[str, Any]]:
        """Return raw Presidio results for error analysis (entity type, span, score)."""
        results = self._analyzer.analyze(text=text, language="en", score_threshold=self.score_threshold)
        return [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
                "text_snippet": text[r.start:r.end][:40],
            }
            for r in results
        ]
