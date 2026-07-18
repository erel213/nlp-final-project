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

## Label Mapping

Presidio entity types are mapped to the project's 5-label DLP schema:

| Label | Triggered by Presidio entity types |
|---|---|
| `PII` | `EMAIL_ADDRESS`, `PHONE_NUMBER`, `PERSON`, `LOCATION`, `IP_ADDRESS`, `URL`, `DATE_TIME`, `NRP`, `US_DRIVER_LICENSE`, + all financial + all health types |
| `financial` | `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `US_BANK_NUMBER`, `US_ITIN`, `US_PASSPORT`, `CRYPTO` |
| `health` | `MEDICAL_LICENSE`, `US_HEALTHCARE_NPI` |
| `confidential` | — (no Presidio recognizer; always 0 in current evaluation) |
| `benign` | 1 when no entity detected above `score_threshold` |

`financial` and `health` are subsets of `PII` — when either fires, `PII=1` as well.

The `financial` types mirror the `FINPII_TYPES` definition from `notebooks/01_data_exploration.ipynb` and `.claude/rules/datasets.md`.

## Usage

```python
from models.rule_based import RuleBasedDetector

detector = RuleBasedDetector(score_threshold=0.4)

# Single text
detector.predict("My card number is 4111 1111 1111 1111")
# → {'benign': 0, 'PII': 1, 'financial': 1, 'health': 0, 'confidential': 0}

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
2. Derives ground-truth labels from `privacy_mask` spans
3. Runs `RuleBasedDetector.predict_batch()` on each `source_text`
4. Computes precision, recall, F1 per label (`benign`, `PII`, `financial`)
5. Plots confusion matrices
6. Shows false-positive / false-negative examples for error analysis
7. Saves metrics to `evaluation/results/rule_based_metrics.json`

## Expected Performance Characteristics

| Strength | Weakness |
|---|---|
| High precision on structured financial data (credit cards, IBANs) — regex patterns are exact | Misses implicit or obfuscated PII (e.g., `my card ends in 1111`) |
| Fast inference — no GPU required, ~thousands of texts/second | High false-positive rate on benign text containing phone-like or date-like strings |
| No training data required | Cannot detect context-dependent leakage (`"My SSN is irrelevant here"` still fires) |
| Transparent and auditable rule set | No coverage for `health` or `confidential` categories in current evaluation |

The precision-recall gap between the rule-based model and fine-tuned transformers quantifies the value of contextual understanding — a key result in the research paper.

## Role in the Ensemble

The rule-based model contributes high-precision signals for structured entities (financial data especially). In the confidence-weighted late-fusion ensemble, its weight is expected to be higher for the `financial` label than for `PII` or `benign`, since BERT/RoBERTa add more value on contextual and implicit leakage.
