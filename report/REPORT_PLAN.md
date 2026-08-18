# Report Plan — SentinelMail Final Report (LaTeX)

Plan for writing the final course report in `report/`, following the course Overleaf
template (`\documentclass{article}` + natbib + graphicx, sections: Introduction /
Methodology / Experimental results / Discussion / Code). Max **8 pages**, PDF.

## Deliverable files

| File | Purpose |
|---|---|
| `report/report.tex` | Main LaTeX source, template structure below |
| `report/references.bib` | BibTeX entries (plain style, natbib) |
| `report/figures/` | Exported PNG/PDF figures from notebooks |
| `report/report.pdf` | Compiled output (≤ 8 pages) |

Title: *SentinelMail: Benchmarking Multi-Model NLP Ensembles for Sensitive-Sentence
Classification*. Author: Erel V. (fill real name + ID). Date line: "Submitted as final
project report for the NLP course, Reichman University (IDC), 2026".

---

## Section-by-section plan

### 1. Introduction (~0.75 page)

- **Motivation:** email DLP — organizations leak PII/financial/confidential data via
  email; rule-based DLP has high false positives and no context understanding.
- **Framing (critical, per comment 003):** the *evaluated task* is multi-label
  sensitive-sentence classification on **ai4privacy/pii-masking-300k**; email DLP is
  the *motivating application only*. State explicitly that results are on synthetic
  sentence-level PII text, not email (external-validity limitation, expanded in
  Discussion).
- **Research questions:**
  - RQ1 — does a multi-model ensemble outperform any single model?
  - RQ2 — does model complementarity vary across sensitivity categories
    (PII / financial / confidential)?
- **Honest scope narrative (grader values process):** briefly note the descoped items
  and why — `health` label (no health entities in the dataset), zero-shot-LLM RQ
  (never implemented under compute constraints), robustness RQ (labels don't flip
  under negation; no contrast set). One tight paragraph — this demonstrates research
  discipline, don't hide it.
- **Novelty positioning (per comment 006 / `documentation/project_novelty.md`):**
  the fusion *mechanism* is standard per-class weighted soft voting — the claimed
  contribution is the **empirical per-category complementarity finding** (RQ2), with
  the ensemble used as a measurement instrument.

#### 1.1 Related Work (~0.5 page)

Sources: `documentation/project_novelty.md` + comment 015 (related work currently
absent from deliverables — this section resolves it for the report).

- Classifier combination: Kittler et al. 1998 (On Combining Classifiers), Kuncheva
  2004 (Combining Pattern Classifiers), Wolpert 1992 (stacking), Jacobs et al. 1991
  (mixture of experts) — position our per-category weighting as an instance of
  per-class weighted soft voting, *not* a new mechanism.
- Models: Devlin et al. 2019 (BERT), Liu et al. 2019 (RoBERTa), Hochreiter &
  Schmidhuber 1997 (LSTM), char-CNN features (Ma & Hovy 2016 or Chiu & Nichols 2016),
  Pennington et al. 2014 (GloVe).
- PII detection / DLP: ai4privacy dataset reference, Microsoft Presidio, and 1–2 DLP
  or PII-classification papers (needs a short literature pass — TODO).

### 2. Methodology (~2 pages)

**Task & data** (source: `data/preprocessing.py`, `data/entity_mapping.py`,
`documentation/preprocessing.md`, notebook 01):
- ai4privacy/pii-masking-300k, English filter; labels **derived from the entity mask**
  via `entity_mapping.py` → 4 labels: `benign`, `PII`, `financial`, `confidential`.
- Be explicit that labels are derived, not independently annotated (comment 004) and
  what that means for what the models can/can't learn.
- Splits (per comment 001 resolution): seeded 10% `train_holdout` slice as DEV for
  selection/weight-learning; the dataset's `validation` split reserved as untouched
  TEST for all reported numbers.
- Class balance table or 1–2 sentences on imbalance (comment 013).

**The four member models** (one short paragraph + design rationale each):
1. **Rule-based** — Presidio/spaCy NER + regex (`models/rule_based/`), zero-trained,
   high-precision structured patterns; note the added confidential recognizers
   (PASSWORD_SECRET, GENERIC_ID, PROPRIETARY_MARK — comment 007 fix).
2. **Bi-LSTM + char-CNN** (`models/bilstm/`) — GloVe 6B 100d frozen embeddings +
   character CNN for obfuscated/structured tokens; 20 epochs, best-val-loss
   checkpointing. ⚠ Report numbers must come from the post-comment-012 retrain
   (real GloVe) — see Blockers.
3. **Fine-tuned BERT** (`models/bert/`) — multi-label head, BCE.
4. **Fine-tuned RoBERTa** (`models/roberta/`).

**Ensemble** (`ensemble/`, `documentation/project_novelty.md`):
- Per-label normalized weighted average of member probabilities; weights fit per
  category on DEV via per-label grid + Nelder–Mead on true F1; per-label decision
  thresholds tuned on DEV (and consistently applied at eval — comment 017 fix).
- State the design choice: linear pooling, *not* stacking/MoE gating, and why
  (interpretability of weights as a complementarity signal).

**Platform / process (template explicitly asks for this):**
- Apple Silicon (MPS) single-machine training; mention concrete MPS workarounds from
  the deepdives (SDPA attention-mask fallback, "Unaligned blit" device-move ordering)
  as technical challenges — graders reward process detail.
- Approximate training times per model (TODO: pull from `run_meta.json` files /
  notebook logs).
- Development process note: iterative internal review produced 17 tracked issues
  (`comments/`) that reshaped splits, metrics, and scope — 1–2 sentences; this is
  strong "process over accuracy" material.

### 3. Experimental results (~2.5 pages)

**Settings:** DEV/TEST protocol as above; metrics: per-label P/R/F1, macro-F1 with
95% bootstrap CI (n=1000, seed 42), AUC-ROC. Decide and state clearly whether macro-F1
includes `benign` (comment 011) and which thresholds are used where (comments 010/017)
— one consistent convention across every table.

**Table 1 — main comparison** (from `evaluation/results/per_class_comparison.csv`,
to be regenerated after retrain queue): per-label F1 + macro-F1 for Majority, Regex,
Presidio rule-based, Bi-LSTM, BERT, RoBERTa, Ensemble. Current (stale) values for
orientation: RoBERTa 0.959 macro-F1, Ensemble 0.966.

**Ensemble detail** (per-label AUC, macro-F1 CI from `ensemble_metrics.json`): folded
into the Table-1 caption and surrounding prose rather than a standalone table, to save
page budget.

**Table 2 — learned per-category weights** (`ensemble/weights.json`) — the RQ2
centerpiece. Narrate the pattern: RoBERTa dominates everywhere; rule-based carries
weight only on financial (0.29) and confidential (0.20); Bi-LSTM contributes on
financial; BERT mostly redundant given RoBERTa.

**RQ1 verdict:** paired-bootstrap delta (ensemble vs best single) with CI and p-value
(`evaluation/ablation.py::ensemble_vs_best_single`, notebook 06). ⚠ Must run on
refreshed weights (comment 009). Be prepared for a *null/marginal* result — the k-fold
fold-0 ablation shows full_ensemble == solo_roberta exactly; "the ensemble does NOT
significantly beat RoBERTa" is a legitimate, reportable finding.

**RQ2 evidence:** leave-one-member-out ablation table (`run_ablation`) + weight table
+ 5-fold CV stability (Bi-LSTM CV done: 0.916 ± 0.010 macro-F1; check status of other
models' folds under `evaluation/results/kfold/`).

**Figures (final choice — the two already exported by notebook 07):**
- F1. `error_overlap.png` — pairwise error-set Jaccard heatmaps (mechanistic "why no
  gain" evidence; has no corresponding table).
- F2. `pr_curves_novelty.png` — ensemble per-category PR curves (operating-point view).
- The originally-planned F1-bars and weight-heatmap figures were dropped: they would
  duplicate Tables 1–2 exactly and cost page budget.

**Error analysis (short):** descriptive buckets from `evaluation/error_analysis.py`
(negation/hypothetical/obfuscation) — explicitly framed as descriptive only, no
research claim (comment 005).

### 4. Discussion (~1 page)

- **Answer RQ1 honestly:** likely "ensemble gain over RoBERTa is small and possibly
  not significant" — frame the insight: with a strong transformer, late fusion adds
  little on this dataset; complementarity shows up per-category, not in the headline.
- **Answer RQ2:** yes — weights and ablation show category-dependent reliance
  (structured categories lean on rule-based/char-level; contextual on transformers).
- **Limitations (each maps to a resolved comment):** synthetic non-email data /
  external validity (003); labels derived from entity masks (004); small per-category
  DEV support → weight instability (016); descoped health/LLM/robustness items.
- **Insights / lessons:** value of internal adversarial review; evaluation-protocol
  pitfalls found and fixed (test-set hygiene, threshold consistency, significance
  testing); "process over accuracy" framing per the template note.

### 5. Code (~2 lines)

Link to the repository (TODO: confirm which remote/URL is the submission link —
**note:** internal Check Point remotes can't be shared externally; if the course needs
a public link, this needs an explicit decision about what gets published where).
One sentence on repo layout + notebooks 01–06 as the reproduction path.

### References (`references.bib`)

kittler1998combining, kuncheva2004combining, wolpert1992stacked, jacobs1991adaptive,
devlin2019bert, liu2019roberta, hochreiter1997lstm, pennington2014glove,
ma2016end (char-CNN), ai4privacy dataset citation, presidio (software citation),
+ 1–2 DLP/PII papers from the related-work pass.

---

## Blockers — RESOLVED (2026-08-16)

The retrain/re-eval queue landed: all artifacts under `evaluation/results/` (incl.
`rq1_verdict.json`, `ablation_novelty.csv`, `complementarity.json`,
`operating_point.csv`, `efficiency.json`, `ensemble_metrics.json`) and
`ensemble/weights.json` are fresh and match `documentation/project_novelty.md`.
All result numbers in `report.tex` were cross-checked against these artifacts.
Remaining TODO in the tex: the repository link in the Code section (submission
location undecided). Compilation/page-count check happens on Overleaf.

## Page budget (8 max)

| Section | Pages |
|---|---|
| Title + Introduction + Related work | 1.25 |
| Methodology | 2.0 |
| Experimental results (incl. tables/figures) | 2.5 |
| Discussion | 1.0 |
| Code + References | 0.75 |
| Slack | 0.5 |
