# Rule-Based DLP Baseline

## Overview

The rule-based baseline is a zero-training model that identifies sensitive content using regex patterns and named-entity recognition (NER). It serves as the lower-bound benchmark against which fine-tuned models (BERT, RoBERTa, Bi-LSTM) are measured in the SentinelMail ensemble.

**Engine:** [Microsoft Presidio](https://github.com/microsoft/presidio) (`presidio-analyzer`) — the open-source industry-standard library for PII detection in production DLP systems. It ships with ~50 built-in recognizers combining compiled regex patterns and a spaCy NER pipeline (`en_core_web_lg`).

## Location

```
models/rule_based/
├── __init__.py           # public: from models.rule_based import RuleBasedDetector
├── detector.py           # RuleBasedDetector class
└── entity_mapping.py     # Presidio entity type → project label mapping
```

Evaluation notebook: `notebooks/02_rule_based_benchmark.ipynb`  
Results output: `evaluation/results/rule_based_metrics.json`

## Label Schema

The project uses a **4-label multi-label schema**: `benign`, `PII`, `financial`, `confidential`.

`health` was originally planned but removed from scope — ai4privacy/pii-masking-300k contains no health-related entity types, making the label untrainable and unevaluable with available data. This is acknowledged as a dataset limitation in the research report.

## Label Mapping

Presidio entity types are mapped to the project's 4-label DLP schema:

| Label | Triggered by Presidio entity types |
|---|---|
| `PII` | `EMAIL_ADDRESS`, `PHONE_NUMBER`, `PERSON`, `LOCATION`, `IP_ADDRESS`, `URL`, `DATE_TIME`, `NRP`, + all financial + all confidential types |
| `financial` | `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `US_BANK_NUMBER`, `US_ITIN`, `CRYPTO` |
| `confidential` | `US_PASSPORT`, `US_DRIVER_LICENSE` |
| `benign` | 1 when no entity detected above `score_threshold` |

`financial` and `confidential` are subsets of `PII` — when either fires, `PII=1` as well.

Ground-truth label mapping (ai4privacy entity types → labels) lives in `data/preprocessing.py`. The Presidio entity types here are the detector-side equivalents.

**Coverage gap:** the ai4privacy `PASS` (password) entity type has no Presidio recognizer — those instances contribute to ground-truth `confidential` positives that the rule-based model cannot detect. This gap is a key finding comparing rule-based vs. learned approaches.

## Usage

```python
from models.rule_based import RuleBasedDetector

detector = RuleBasedDetector(score_threshold=0.4)

# Single text
detector.predict("My passport number is A12345678.")
# → {'benign': 0, 'PII': 1, 'financial': 0, 'confidential': 1}

# Batch
detector.predict_batch(["Call me at 415-555-1234", "Hello, hope you are well."])

# Explain detections (for error analysis)
detector.explain("Contact john.doe@example.com")
# → [{'entity_type': 'EMAIL_ADDRESS', 'start': 8, 'end': 30, 'score': 1.0, 'text_snippet': 'john.doe@example.com'}]
```

## Setup

Install dependencies (one-time):

```bash
pip install presidio-analyzer spacy
python -m spacy download en_core_web_lg
```

## Evaluation

Run `notebooks/02_rule_based_benchmark.ipynb` end-to-end. It:

1. Loads the English validation split of `ai4privacy/pii-masking-300k` (~8k rows)
2. Derives ground-truth labels from `privacy_mask` spans via `data.preprocessing.build_labels`
3. Runs `RuleBasedDetector.predict_batch()` on each `source_text`
4. Computes precision, recall, F1 per label (`benign`, `PII`, `financial`, `confidential`)
5. Plots confusion matrices
6. Shows false-positive / false-negative examples for error analysis
7. Saves metrics to `evaluation/results/rule_based_metrics.json`

## Expected Performance Characteristics

| Strength | Weakness |
|---|---|
| High precision on structured financial data (credit cards, IBANs) — regex patterns are exact | Misses implicit or obfuscated PII (e.g., `my card ends in 1111`) |
| Fast inference — no GPU required, ~thousands of texts/second | High false-positive rate on benign text containing phone-like or date-like strings |
| No training data required | Cannot detect context-dependent leakage (`"My SSN is irrelevant here"` still fires) |
| Transparent and auditable rule set | No coverage for `confidential` passwords (PASS entities) — structural gap vs. learned models |

The precision-recall gap between the rule-based model and fine-tuned transformers quantifies the value of contextual understanding — a key result in the research paper.

## Role in the Ensemble

The rule-based model contributes high-precision signals for structured entities (financial data especially). In the confidence-weighted late-fusion ensemble, its weight is expected to be higher for the `financial` label than for `PII` or `benign`, since BERT/RoBERTa add more value on contextual and implicit leakage.
