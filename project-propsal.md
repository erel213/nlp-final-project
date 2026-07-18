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

# **Project Proposal: Long-Range Narrative Contradiction Detection in Hebrew using Large Language Models**

---

# **Project Title & Problem Statement**

**Title:**  
 **Beyond Sentence Pairs: Benchmarking Large Language Models for Long-Range Narrative Contradiction Detection in Hebrew**

**Problem Statement:**  
 Natural Language Inference (NLI) has traditionally focused on identifying contradictions between pairs of isolated sentences. However, real-world documents such as stories, news articles, legal documents, and conversations often contain contradictions that emerge only after tracking information across many paragraphs.

Recent Large Language Models (LLMs) demonstrate impressive reasoning abilities, but it remains unclear whether they can consistently detect contradictions that require maintaining long-range context throughout an entire narrative—especially in Hebrew, a language with considerably fewer evaluation resources than English.

This project investigates the ability of modern LLMs to detect long-range narrative contradictions in Hebrew stories and analyzes which types of contradictions remain challenging.

---

# **Detailed Description of Idea and Innovation Highlights**

## **Core Idea**

We construct a benchmark consisting of Hebrew stories in two versions: one internally consistent and one containing carefully inserted long-range contradictions. We then compare several state-of-the-art LLMs on their ability to identify whether a contradiction exists, locate it, and explain the reasoning behind their decision.

---

## **Innovation Highlights**

### **1\. Hebrew-first benchmark**

Most existing contradiction datasets are written in English and focus on sentence-level inference.

We propose one of the first benchmarks dedicated to long-range contradiction detection in Hebrew narratives.

---

### **2\. Long-context reasoning**

Instead of comparing two isolated sentences, models must remember facts introduced much earlier in the story and determine whether later events remain logically consistent.

This better reflects realistic reading-comprehension scenarios.

---

### **3\. Multiple contradiction categories**

Rather than evaluating a single contradiction type, we generate several categories, including:

* Character identity contradictions  
* Timeline inconsistencies  
* Object ownership contradictions  
* Location inconsistencies  
* Preference and personality contradictions  
* Cause-effect contradictions

This allows fine-grained analysis of model weaknesses.

---

### **4\. Explainable evaluation**

Besides predicting whether a contradiction exists, models are evaluated on whether they correctly identify:

* the contradictory statements,  
* the reasoning behind the contradiction,  
* and the contradiction category.

This enables qualitative comparison beyond simple accuracy.

---

# **Implementation Steps**

## **1\. Hebrew Dataset Construction**

Create a collection of Hebrew stories.

For each original story:

* Create a consistent version.  
* Create one or more contradictory versions by modifying only a small number of facts while keeping the rest of the story unchanged.

Contradiction types include:

* Character death followed by later reappearance  
* Inconsistent timeline  
* Changing ownership of an object  
* Conflicting preferences or personal traits  
* Location inconsistencies  
* Impossible cause-and-effect relationships

---

## **2\. Baseline Models**

Evaluate multiple state-of-the-art LLMs, for example:

* GPT-5  
* Claude  
* Gemini  
* Llama 3

---

## **3\. Prompt Design**

Experiment with different prompting strategies:

* Zero-shot prompting  
* Few-shot prompting  
* Chain-of-Thought prompting  
* Structured reasoning prompts

Compare whether explicit reasoning improves contradiction detection.

---

## **4\. Evaluation & Analysis**

Evaluation metrics:

* Accuracy  
* Precision  
* Recall  
* F1-score

Additional analyses:

* Performance by contradiction category  
* Effect of contradiction distance (number of paragraphs separating conflicting facts)  
* Effect of story length  
* Quality of generated explanations  
* Error analysis

---

## **5\. Report & Presentation**

Summarize quantitative and qualitative findings and discuss what they reveal about long-context reasoning capabilities in Hebrew.

---

# **Methodology & Datasets**

## **Datasets**

### 

| Dataset | Source | Role |
| :---- | :---- | :---- |
| Self-generated Hebrew stories | Generated manually and with LLM assistance | Main evaluation benchmark |
| Human-validated subset | Manually reviewed | Ground-truth quality assurance |
| Existing English contradiction datasets (optional) | Public datasets | Reference for comparison |

### 

## **Datasets**

| Dataset | Source | Role |
| ----- | ----- | ----- |
| Self-generated Hebrew stories | Generated manually and with LLM assistance | Main evaluation benchmark |
| Human-validated subset | Manually reviewed | Ground-truth quality assurance |
| Existing English contradiction datasets (optional) | Public datasets | Reference for comparison |

---

## **Models**

| Model | Type | Role |
| ----- | ----- | ----- |
| GPT-5 | Large Language Model | Long-context reasoning |
| Claude | Large Language Model | Performance comparison |
| Gemini | Large Language Model | Performance comparison |
| Llama 3 | Open-source LLM | Open-source baseline |
| Qwen (optional) | Open-source LLM | Additional comparison |

---

# **Research Questions**

1. How accurately can modern LLMs detect long-range narrative contradictions in Hebrew?  
2. Which categories of contradictions are the most challenging for current LLMs?  
3. Does explicit reasoning (Chain-of-Thought) improve contradiction detection performance?  
4. How does model performance change as the distance between contradictory facts increases?  
5. Does story length significantly affect contradiction detection accuracy?

---

# **Notes on Resources & Feasibility**

* The benchmark can contain approximately 300–1,000 Hebrew stories, since the project evaluates reasoning capabilities rather than training large supervised models.  
* Stories can be generated with LLM assistance and manually reviewed to ensure high quality and naturalness.  
* Evaluation requires only API access to existing LLMs, making the project computationally inexpensive.  
* The project addresses an underexplored area in Hebrew NLP, where there are currently very few benchmarks for long-context reasoning and contradiction detection.  
* The resulting Hebrew benchmark could become a reusable evaluation resource for future research on Hebrew language understanding and reasoning.

