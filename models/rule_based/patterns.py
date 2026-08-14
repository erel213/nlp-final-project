"""Custom Presidio recognizers for the rule-based DLP baseline.

Implements the confidential-category recognizers specced in
`.claude/rules/model-rule-based.md` (password/secret patterns, generic
passport / ID / driver-license numbers, and a proprietary-vocabulary matcher).

Rationale (comment 007): stock Presidio ships only US-specific
`US_PASSPORT` / `US_DRIVER_LICENSE` recognizers and has NO recognizer for
passwords/secrets. ai4privacy `PASS`/`PASSPORT`/`IDCARD`/`DRIVERLICENSE`
entities are generic/international, so the rule-based model scored a
structural 0.0 F1 on `confidential`. These recognizers give it a genuine
shot at the category while staying fully zero-trained (ADR-002).

All recognizers emit custom Presidio entity types that are mapped to the
`confidential` label via `PRESIDIO_CONFIDENTIAL_ENTITIES` in
`data/entity_mapping.py`:

    PASSWORD_SECRET  — password / API-key / token / secret mentions
    GENERIC_ID       — generic passport / national-ID / driver-license numbers
    PROPRIETARY_MARK — proprietary / classification-marker vocabulary
"""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer

# Custom entity types emitted by the recognizers below.
PASSWORD_SECRET = "PASSWORD_SECRET"
GENERIC_ID = "GENERIC_ID"
PROPRIETARY_MARK = "PROPRIETARY_MARK"

CUSTOM_CONFIDENTIAL_ENTITIES: frozenset[str] = frozenset(
    {PASSWORD_SECRET, GENERIC_ID, PROPRIETARY_MARK}
)


def build_password_recognizer() -> PatternRecognizer:
    """Password / API-key / token / secret mentions.

    Patterns per model-rule-based.md "Confidential Detection":
      - password/passwd/pwd/passphrase followed by a value
      - api-key/token/secret/bearer followed by a high-entropy value
    Both require a *value* after the keyword to avoid firing on the bare word
    (e.g. "please reset your password" should NOT trigger confidential).
    """
    patterns = [
        Pattern(
            name="password_kv_delimited",
            # keyword + explicit delimiter + value  (e.g. "password: r]iD1#8")
            regex=r"\b(?:password|passwd|pwd|passphrase)\b\s*[:=]\s*\S{3,}",
            score=0.7,
        ),
        Pattern(
            name="password_value_entropy",
            # keyword + whitespace + a value containing a digit or symbol.
            # Requiring a non-alpha char avoids firing on prose like
            # "reset your password soon" while still catching "password r]iD1#8".
            regex=r"\b(?:password|passwd|pwd|passphrase)\b\s+\S*[\d!-/:-@\[-`{-~]\S*",
            score=0.7,
        ),
        Pattern(
            name="api_key_token",
            regex=r"\b(?:api[-_ ]?key|token|secret|bearer)\b\s*[:=]?\s*[A-Za-z0-9+/=_\-]{12,}",
            score=0.7,
        ),
    ]
    return PatternRecognizer(
        supported_entity=PASSWORD_SECRET,
        patterns=patterns,
        supported_language="en",
    )


def build_generic_id_recognizer() -> PatternRecognizer:
    """Generic (non-US) passport / national-ID / driver-license numbers.

    ai4privacy examples: IDCARD "KB90324ER", passport/DL alphanumeric codes.
    Uses a document-number shape plus context keywords so that a bare
    alphanumeric token does not fire on its own.
    """
    patterns = [
        # explicit context keyword followed by an alphanumeric document number
        Pattern(
            name="id_with_context",
            regex=(
                r"\b(?:passport|driver'?s?\s+licen[sc]e|driving\s+licen[sc]e|"
                r"id\s*card|identity\s*card|national\s*id|"
                r"id\s+(?:number|no\.?|#)|dl#?)\b"
                r"[\s:#=.]*(?:no\.?\s*)?[A-Z0-9]{6,12}\b"
            ),
            score=0.6,
        ),
        # canonical document-number shape: 1-2 leading letters, 4-9 digits,
        # optional trailing letters (e.g. "KB90324ER", "L898902C3", "AB1234567")
        Pattern(
            name="doc_number_shape",
            regex=r"\b[A-Z]{1,2}\d{4,9}[A-Z]{0,2}\b",
            score=0.4,
        ),
    ]
    return PatternRecognizer(
        supported_entity=GENERIC_ID,
        patterns=patterns,
        supported_language="en",
    )


def build_proprietary_recognizer() -> PatternRecognizer:
    """Proprietary / classification-marker vocabulary.

    Implements ADR-005's confidential vocabulary. The spec calls for a spaCy
    PhraseMatcher; for fixed multi-word markers a case-insensitive Presidio
    ``Pattern`` is functionally equivalent and keeps every confidential signal
    inside the single AnalyzerEngine pipeline (see comment 007 note). Deviation
    is intentional and documented.
    """
    phrases = [
        r"confidential",
        r"internal\s+only",
        r"do\s+not\s+distribute",
        r"proprietary",
        r"trade\s+secret",
        r"restricted",
        r"classified",
    ]
    patterns = [
        Pattern(
            name=f"proprietary_{i}",
            regex=rf"\b{phrase}\b",
            score=0.5,
        )
        for i, phrase in enumerate(phrases)
    ]
    return PatternRecognizer(
        supported_entity=PROPRIETARY_MARK,
        patterns=patterns,
        supported_language="en",
    )


def build_confidential_recognizers() -> list[PatternRecognizer]:
    """All custom confidential-category recognizers, ready to register."""
    return [
        build_password_recognizer(),
        build_generic_id_recognizer(),
        build_proprietary_recognizer(),
    ]
