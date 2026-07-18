# Data Preprocessing

## Overview

The `data/` module is the single source of truth for dataset loading, entity-type → label mapping, and multi-label construction. Every model in the project (rule-based, BERT, RoBERTa, Bi-LSTM, ensemble) must derive its ground-truth labels from this module — never inline its own mapping.

```
data/
├── entity_mapping.py   # all entity type sets (both vocabularies)
├── preprocessing.py    # parse_mask, build_labels, load_split, LABEL_COLS
└── __init__.py
```

---

## Label Schema

The project uses a **4-label multi-label schema**. An email can belong to more than one category simultaneously.

| Label | Description |
|---|---|
| `benign` | No sensitive entities detected |
| `PII` | Any personally identifiable information present |
| `financial` | Financial identifiers present (card issuer, SSN) |
| `confidential` | Secrets or government identity documents present |

`benign` is mutually exclusive with the other three — if any entity is found, `benign = 0`. `financial` and `confidential` are subsets of `PII`: whenever either fires, `PII = 1` as well.

`health` was removed from scope: `ai4privacy/pii-masking-300k` contains no health-related entity types, making the label untrainable and unevaluable.

---

## Entity Mapping (`data/entity_mapping.py`)

This file holds all entity type → label mappings for both vocabularies used in the project. **Do not define entity sets anywhere else.**

### ai4privacy vocabulary (ground-truth)

These are the `label` values found in the dataset's `privacy_mask` field.

| Variable | Entity types | Label |
|---|---|---|
| `FINANCIAL_ENTITY_TYPES` | `CARDISSUER`, `SOCIALNUMBER` | `financial` |
| `CONFIDENTIAL_ENTITY_TYPES` | `PASS`, `PASSPORT`, `IDCARD`, `DRIVERLICENSE` | `confidential` |

All 27 entity types in the dataset contribute to `PII`. Anything not in the financial or confidential sets is purely `PII`.

**Entity type clarifications:**
- `CARDISSUER` — credit/debit card issuer name (e.g. "Diners Club International"), not a card number
- `SOCIALNUMBER` — social security / national insurance number (e.g. "996 076 6460")
- `PASS` — confirmed **password** (values like `r]iD1#8`, `Be~o}.zq8^1"`) — distinct from `PASSPORT`
- `PASSPORT` — passport document number
- `IDCARD` — national ID card number (e.g. "KB90324ER")
- `DRIVERLICENSE` — driver's license number

### Presidio vocabulary (rule-based model predictions)

These are the entity types emitted by Microsoft Presidio's `AnalyzerEngine`, used only by `models/rule_based/detector.py`.

| Variable | Entity types | Label |
|---|---|---|
| `PRESIDIO_FINANCIAL_ENTITIES` | `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `US_BANK_NUMBER`, `US_ITIN`, `CRYPTO` | `financial` |
| `PRESIDIO_CONFIDENTIAL_ENTITIES` | `US_PASSPORT`, `US_DRIVER_LICENSE` | `confidential` |
| `PRESIDIO_PII_ENTITIES` | Union of all above + `EMAIL_ADDRESS`, `PHONE_NUMBER`, `PERSON`, `LOCATION`, `IP_ADDRESS`, `URL`, `DATE_TIME`, `NRP` | `PII` |

**Coverage gap:** `PASS` (password) has no Presidio recognizer. Rows with password entities contribute to ground-truth `confidential` positives that the rule-based model structurally cannot detect. This is a known gap documented in the research findings.

---

## API (`data/preprocessing.py`)

### `parse_mask(mask) → list[dict]`

Normalises a `privacy_mask` value from the HuggingFace dataset into a plain Python list of entity dicts, regardless of how it was stored (list, numpy array, JSON string, or None).

```python
from data.preprocessing import parse_mask

entities = parse_mask(row["privacy_mask"])
# → [{'value': 'Diners Club', 'start': 10, 'end': 21, 'label': 'CARDISSUER'}, ...]
```

### `build_labels(privacy_mask) → dict[str, int]`

Maps a single `privacy_mask` value to the 4-label dict. This is the function all models should use to derive ground-truth labels.

```python
from data.preprocessing import build_labels

build_labels([{"label": "CARDISSUER", "value": "Visa", "start": 0, "end": 4}])
# → {"benign": 0, "PII": 1, "financial": 1, "confidential": 0}

build_labels([{"label": "PASS", "value": "r]iD1#8", "start": 0, "end": 7}])
# → {"benign": 0, "PII": 1, "financial": 0, "confidential": 1}

build_labels([])
# → {"benign": 1, "PII": 0, "financial": 0, "confidential": 0}
```

### `load_split(split, language="English") → pd.DataFrame`

Loads a dataset split, filters by language, and attaches the 4 label columns. Returns the full HuggingFace DataFrame augmented with `benign`, `PII`, `financial`, `confidential` columns.

```python
from data.preprocessing import load_split

train_df = load_split("train")
val_df   = load_split("validation")
```

### `LABEL_COLS`

The canonical ordered list of label column names. Use this everywhere instead of hardcoding the list.

```python
from data.preprocessing import LABEL_COLS
# → ["benign", "PII", "financial", "confidential"]
```

---

## Dataset Split Statistics (English only)

| Split | Total rows | benign | PII | financial | confidential |
|---|---|---|---|---|---|
| train | 29,908 | 3,482 (11.6%) | 26,426 (88.4%) | — | — |
| validation | 7,946 | 908 (11.4%) | 7,038 (88.6%) | 1,529 (19.2%) | 3,632 (45.7%) |

`financial` and `confidential` counts for the training split were not computed at time of writing — run `notebooks/01_data_exploration.ipynb` to populate them.

`financial` and `confidential` are subsets of `PII`, so their percentages are relative to total rows, not to PII rows.

---

## Applying Labels in a New Model

Always import from `data.preprocessing` — never define your own entity type sets or label logic:

```python
from data.preprocessing import build_labels, load_split, LABEL_COLS

# Option A — load the full split (recommended for training scripts)
df = load_split("train")
y = df[LABEL_COLS].values  # shape (N, 4)

# Option B — apply labels to a pre-loaded DataFrame
import pandas as pd
label_df = df["privacy_mask"].apply(build_labels).apply(pd.Series)
```

For the rule-based model specifically, also import from `data.entity_mapping`:

```python
from data.entity_mapping import (
    PRESIDIO_FINANCIAL_ENTITIES,
    PRESIDIO_CONFIDENTIAL_ENTITIES,
    PRESIDIO_PII_ENTITIES,
)
```
