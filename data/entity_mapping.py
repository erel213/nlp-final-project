# Single source of truth for entity type → DLP label mappings.
#
# Two vocabularies are represented here — both map to the same 4-label schema
# {benign, PII, financial, confidential}:
#
#   ai4privacy  — entity types used in the dataset's privacy_mask field
#                 (ground-truth labels for training and evaluation)
#   Presidio    — entity types produced by the Presidio analyzer
#                 (predictions from the rule-based model)

# ---------------------------------------------------------------------------
# ai4privacy vocabulary  (ground-truth)
# ---------------------------------------------------------------------------

# CARDISSUER = card issuer name (e.g. "Diners Club International")
# SOCIALNUMBER = social security / national insurance number (e.g. "996 076 6460")
FINANCIAL_ENTITY_TYPES: frozenset[str] = frozenset({
    "CARDISSUER",
    "SOCIALNUMBER",
})

# PASS confirmed = password (values: "q4R\\", "r]iD1#8") — distinct from PASSPORT.
# PASS (password) has no Presidio recognizer → structural recall gap for rule-based model.
CONFIDENTIAL_ENTITY_TYPES: frozenset[str] = frozenset({
    "PASS",
    "PASSPORT",
    "IDCARD",
    "DRIVERLICENSE",
})

# ---------------------------------------------------------------------------
# Presidio vocabulary  (rule-based model predictions)
# ---------------------------------------------------------------------------

PRESIDIO_FINANCIAL_ENTITIES: frozenset[str] = frozenset({
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_ITIN",
    "CRYPTO",
})

PRESIDIO_CONFIDENTIAL_ENTITIES: frozenset[str] = frozenset({
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
})

PRESIDIO_PII_ENTITIES: frozenset[str] = (
    PRESIDIO_FINANCIAL_ENTITIES
    | PRESIDIO_CONFIDENTIAL_ENTITIES
    | frozenset({
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "PERSON",
        "LOCATION",
        "IP_ADDRESS",
        "URL",
        "DATE_TIME",
        "NRP",
    })
)
