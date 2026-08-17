# Project Proposal: Email DLP — Sensitive Data Detection via Multi-Model NLP Ensemble

## Project Title & Problem Statement

**Title:** SentinelMail: Benchmarking Multi-Model NLP Ensembles for Email Data Loss Prevention (DLP)

**Problem Statement:** Organizations face a persistent threat of sensitive data leakage through email — employees may accidentally (or deliberately) send emails containing PII, financial data, medical records, or trade secrets. Traditional rule-based DLP systems (regex patterns, keyword lists) have high false-positive rates and fail to understand context. For example, "My SSN is irrelevant here" should not trigger a PII alert, but a rigid pattern matcher will flag it.

This project investigates whether combining multiple NLP models — rather than relying on a single classifier — can meaningfully improve sensitive content detection accuracy while reducing false positives in email DLP scenarios.

---

## Detailed Description of Idea and Innovation Highlights

### Core Idea

We benchmark and compare several NLP models on the task of email sensitive-data classification, then design an ensemble strategy that fuses their predictions. The classification is multi-label: an email may contain PII, financial data, health information, or be fully benign.

### Innovation Highlights

1. **Multi-label sensitive-category classification** — rather than a binary "sensitive / not sensitive" label, we distinguish between sensitivity *types* (PII, financial, health, confidential), which is more actionable for DLP systems.  
     
2. **Context-aware vs. pattern-aware fusion** — we combine models that excel at different signals:  
     
   - **Fine-tuned BERT / RoBERTa** — contextual understanding (catches implicit leakage, negations, hypotheticals)  
   - **Bi-LSTM with character-level features** — strong at detecting obfuscated patterns (e.g., `4155552368` vs `415-555-2368`)  
   - **Rule-augmented model** — a lightweight regex/NER baseline that captures high-precision known patterns (credit card numbers, SSNs)

   

3. **Novelty in ensemble design** — we propose a *confidence-weighted late fusion* mechanism where each model's weight is tuned per sensitivity category, since BERT may outperform the rule model on implicit PII but underperform on structured financial data.

---

## Implementation Steps

1. **Dataset Construction**  
     
   - Start with the Enron Email Dataset (public, \~500k emails)  
   - Inject synthetic sensitive content (PII, financials, health data) using templates  
   - Annotate a validation set manually (crowdsource-style annotation with inter-annotator agreement)  
   - Label categories: `[benign, PII, financial, health, confidential]`

   

2. **Baseline Models**  
     
   - Rule-based: regex \+ spaCy NER pipeline  
   - Fine-tuned `bert-base-uncased` on training set (multi-label classification head)  
   - Fine-tuned `roberta-base` on training set  
   - Bi-LSTM with GloVe \+ character embeddings

   

3. **Ensemble Design**  
     
   - Late fusion: weighted average of model probability outputs  
   - Learn category-specific weights via held-out validation set  
   - Analyze contribution of each model per category

   

4. **Evaluation & Analysis**  
     
   - Metrics: F1 (macro & per-class), precision, recall, AUC-ROC  
   - Error analysis: false positives (negations, hypotheticals), false negatives (obfuscated data)  
   - Ablation study: remove each model from ensemble to measure marginal contribution

   

5. **Report & Presentation**

---

## Methodology & Datasets

### Datasets

| Dataset | Source | Role |
| :---- | :---- | :---- |
| Enron Email Corpus | [CMU](https://www.cs.cmu.edu/~enron/) / [Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) | Base email text (benign emails) |
| Synthetic PII injection | Self-generated templates | Controlled sensitive content |
| Manually annotated validation set | \~500 emails, manually labeled | Ground truth for evaluation |
| MIMIC-III Clinical Notes (optional) | [PhysioNet](https://physionet.org/content/mimiciii/1.4/) | Cross-domain health category test |

### Models

| Model | Type | Role |
| :---- | :---- | :---- |
| `bert-base-uncased` | Transformer encoder | Contextual classification |
| `roberta-base` | Transformer encoder | Contextual classification (stronger baseline) |
| Bi-LSTM \+ char-CNN | Recurrent | Pattern-sensitive, low resource |
| spaCy NER \+ regex rules | Rule-based | High-precision baseline |
| LLaMA-3-8B / Claude Haiku | LLM, zero-shot | Generalization probe |

### Research Questions

1. Does a multi-model ensemble outperform any single model for email DLP classification?  
2. Does model complementarity vary across sensitivity categories (PII vs. financial vs. health)?  
3. Can a zero-shot LLM substitute for fine-tuned models when labeled data is scarce?  
4. How robust is each approach to context manipulation (negations, hypotheticals, obfuscation)?

---

## Notes on Resources & Feasibility

- Fine-tuning BERT/RoBERTa can be done on a single GPU (Colab T4 sufficient for the email classification task with \~50k training examples).  
- LLaMA-3-8B inference via Ollama locally or via API; Claude Haiku is inexpensive via API.  
- Enron dataset is freely available and widely used — no licensing issues.  
- Dataset construction is the main manual effort; synthetic injection reduces annotation burden significantly.
