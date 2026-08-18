# Project Novelty: What SentinelMail Contributes

This note states, honestly and precisely, what SentinelMail contributes as a research
project — not just the ensemble fusion method, but the project-level finding, its
mechanistic explanation, and the evaluation methodology used to reach it. Every
quantitative claim here is produced by `notebooks/07_novelty_analysis.ipynb` from member
probabilities computed fresh via `ensemble.compute_probas` (no on-disk cache), with
artifacts under `evaluation/results/` (`rq1_verdict.json`, `rq1_pairwise.csv`,
`ablation_novelty.csv`, `operating_point.csv`, `efficiency.json`) and figures under
`report/figures/`.

Scope note: this doc broadened from a fusion-method-only positioning to the project's
overall novelty. It still disclaims the fusion *mechanism* as standard (see "The fusion
mechanism is not novel"); the contribution lies elsewhere. Broader related-work grounding
for DLP and PII classification is tracked separately (comment 015); this doc
cross-references but does not duplicate it.

## Research questions (as answered by notebook 07)

The project's two research questions (mirroring `CLAUDE.md`), restated so each maps to a
concrete measurement in `notebooks/07_novelty_analysis.ipynb`:

- **RQ1 — Does a multi-model ensemble outperform the best single model for
  sensitive-content classification (ai4privacy; motivating application: email DLP)?**
  Measured by a paired-bootstrap significance test on the headline macro-F1 delta
  (ensemble − best single), tuned thresholds applied identically to every system, plus a
  Holm-corrected pairwise panel against *every* member (notebook §1). **Answer: yes, but
  small** — the ensemble beats the best single model (RoBERTa) with a delta whose 95% CI
  excludes zero, but the margin is a few tenths of a macro-F1 point.

- **RQ2 — Does the ensemble's reliance on its members vary across sensitivity categories
  (PII vs. financial vs. confidential)?** Measured by the learned per-category weights and
  the leave-one-member-out ablation (notebook §2). **Answer: a qualified no** — the learned
  weights *look* like per-category specialization, but the ablation shows the ensemble's
  real reliance is concentrated on RoBERTa almost everywhere, not cleanly differentiated by
  category (see caveats below).

The honest, defensible novelty of the project is the *combination* of these answers plus
the machinery that produced them, described next.

## The contribution, in three parts

1. **An empirical finding, rigorously significance-gated.** On multi-label
   sensitive-sentence classification (ai4privacy, DLP-motivated), a per-category weighted
   ensemble of four heterogeneous NLP members (BERT, RoBERTa, Bi-LSTM+char-CNN,
   spaCy/Presidio+regex) **does** beat a single fine-tuned RoBERTa on headline macro-F1 —
   the delta's 95% CI excludes zero under a paired bootstrap, and it survives a
   Holm–Bonferroni correction against *every* member, not just the hardest comparison. But
   the gain over the transformers is **practically small** (≤ a few tenths of a point).
   The contribution is the *rigor*: significance-gated (paired bootstrap, not eyeballed
   deltas), multi-comparison (Holm across the whole panel), and reported with effect size
   alongside significance so a tiny-but-real gain at N≈8k is never dressed up as a large one.

2. **A mechanistic explanation for *why the gain is small*.** We do not just report a small
   delta; we explain it. The leave-one-out ablation shows the ensemble's reliance is
   concentrated on **RoBERTa**: removing RoBERTa drops macro-F1 ~1.5 points, while removing
   any other member costs ≤0.5. Because the fused system leans so heavily on the single
   best member, the headroom to beat that member on its own is inherently small — which is
   exactly the RQ1 result. The value the other members add is mostly at the margins and on
   the weaker categories, not a broad lift over RoBERTa.

3. **A reusable, honesty-first evaluation methodology.** The project packages a small
   harness for judging ensemble claims without overclaiming: a paired-bootstrap
   significance gate (`ensemble_vs_best_single`), a Holm-corrected multi-comparison panel
   (`ensemble_vs_each_model`), a leave-one-out ablation (`run_ablation`), a DLP-relevant
   operating-point view (`precision_at_recall` / `operating_point_table`), and an
   efficiency/cost accounting. Each lives in the `evaluation` package and is exercised by
   notebook 07. The methodological point — *confirm an ensemble claim with an ablation AND
   a significance test, correct for multiple comparisons, report effect size next to
   p-values, and judge at the operating point the application will actually use, not at
   F1-optimal thresholds* — is itself a transferable takeaway.

## In plain terms (read this first)

**What we built.** Four models that spot sensitive text (PII, financial, confidential):
two large language models (BERT, RoBERTa), one smaller neural network (Bi-LSTM), and one
rule-based checker (spaCy/Presidio + regex). Then an **ensemble** that asks all four and
blends their answers, with the blend weighted *per category*.

**What we hoped.** Different models are good at different things, so a smart per-category
blend should beat any single model — and mapping "who's good at what" would be a finding
in itself.

**What we found.** Tested fairly on held-out data, **the ensemble does beat the best single
model** (RoBERTa): 0.974 vs 0.972 headline macro-F1. The gap is statistically real (its
confidence interval excludes zero, p = 0.006) but *small*. Against the whole panel of
members the ensemble wins every pairwise comparison after a multiple-comparison correction,
with the margin growing as the member gets weaker (tiny over RoBERTa and BERT, large over
Bi-LSTM and the rule-based baseline).

**Does the per-category story hold up?** Mostly no. The fitted weights *look* like
specialization (RoBERTa heavy on benign/PII/confidential; financial spread across three
members). But when we remove members one at a time, only **RoBERTa** matters much: drop it
and the score falls ~1.5 points; drop anyone else and it barely moves (≤0.5). PII is
unaffected by removing *any* single member. So the ensemble's real reliance is concentrated
on one model, not cleanly split by category.

**Why doesn't the ensemble win by more?** Because it leans so heavily on RoBERTa. When the
fused system is essentially RoBERTa-plus-a-little, the room to beat RoBERTa alone is small.
That is the same fact seen from two angles — the RQ2 ablation (reliance concentrated on
RoBERTa) explains the RQ1 result (small gain over RoBERTa).

**Anywhere it helps more clearly?** At the high-recall operating point DLP uses (recall ≥
99%), the ensemble has the best per-category precision on all three *informative* categories
(PII, financial, confidential); it loses only benign to BERT. This is a modest edge the
headline metric understates — but those thresholds were picked on the test set itself
(optimistic) and the edge is not significance-tested, so we report it as a supporting
qualifier, not an independent win.

**So what's the contribution?** A rigorously significance-gated, multi-comparison finding
(a real but small ensemble gain over every member), a clear explanation of why the gain is
small (reliance concentrated on RoBERTa rather than clean category-wise complementarity),
and a reusable, honesty-first way to evaluate ensemble claims — plus the caution that
plausible-looking learned weights and F1-optimal thresholds can both mislead, and that
statistical significance at large N does not imply operational significance.

**Terms used below:** *Macro-F1* — headline score averaged over PII/financial/confidential.
*Held-out/test* — data the models never trained on. *Ablation* — remove a model and
re-measure. *Paired bootstrap* — resamples the same rows for both systems to test whether
a gap is real (CI excludes 0) or noise. *Holm correction* — a step-down adjustment that
controls the family-wise error rate across the four pairwise tests.

## The fusion mechanism is not novel (explicitly disclaimed)

The ensemble combines members by a per-label normalized weighted average of their
probabilities:

    score_L = ( Σ_m w_{m,L} · p_{m,L} ) / ( Σ_m w_{m,L} )

with a separate weight vector fit per category to maximize per-label F1 on the
selection/dev split. This is **standard per-class weighted soft voting**. We do not claim
the fusion rule, the per-class weighting, or the weight-fitting as new — each is covered
by prior work:

- **Weighted soft voting / linear opinion pooling** (Kittler et al., 1998) — averaging
  class posteriors with tuned weights is classical.
- **Per-class / label-wise classifier combination** (Kuncheva, 2004) — a different
  combiner weight per class is long established; our per-category weights are an instance.
- **Stacking / stacked generalization** (Wolpert, 1992) — a learned meta-classifier is
  strictly more expressive; we deliberately forgo it (small dev split; comment 016).
- **Mixture-of-experts gating** (Jacobs et al., 1991) — input-conditioned routing; our
  weights are input-independent, a simpler non-gated special case.

"We tune the weight separately per category" alone would be subsumed by per-class
classifier combination and would not carry a novelty claim. The contribution is the
empirical finding and methodology above, not the combiner.

## Demonstrated evidence

**RQ1 — a small but significant ensemble gain** (held-out test, tuned per-label thresholds
applied identically to every system; comment 017 / comment 010; `rq1_verdict.json`):

    Ensemble macro-F1 = 0.97441   vs   best single (RoBERTa) = 0.97174
    delta = +0.00264   95% CI [+0.00061, +0.00478]   paired-bootstrap p = 0.006
    VERDICT: SIGNIFICANT — delta CI excludes 0.

The ensemble beats RoBERTa alone, but by a few tenths of a point. Significance and effect
size are reported together on purpose: at N≈8k the paired bootstrap resolves tiny-but-real
differences, so "significant" here means *reliably positive*, not *large*.

**RQ1 pairwise panel — significant vs. every member under Holm** (`rq1_pairwise.csv`).
Ensemble vs. each member on the same resampled TEST rows and DEV-tuned thresholds,
Holm-corrected across the family of four tests:

| model      | model macro-F1 | delta    | 95% CI                 | Holm p | significant |
|------------|----------------|----------|------------------------|--------|-------------|
| roberta    | 0.972          | +0.00264 | [+0.00061, +0.00478]   | 0.006  | yes         |
| bert       | 0.961          | +0.01318 | [+0.01033, +0.01618]   | 0.000  | yes         |
| bilstm     | 0.943          | +0.03164 | [+0.02806, +0.03537]   | 0.000  | yes         |
| rule_based | 0.802          | +0.17213 | [+0.16447, +0.17985]   | 0.000  | yes         |

Every member's delta CI excludes zero and survives the correction, so the win is not an
artifact of picking one comparison. Magnitudes scale with member strength: a few tenths of
a point over the transformers, much larger over Bi-LSTM and the rule-based baseline.

**RQ2 — reliance is concentrated on RoBERTa, not cleanly per-category.**
Learned per-category weights (rows = model, cols = label):

| model      | benign | PII  | financial | confidential |
|------------|--------|------|-----------|--------------|
| bert       | 0.10   | 0.10 | 0.10      | 0.00         |
| roberta    | 0.70   | 0.70 | 0.30      | 0.60         |
| bilstm     | 0.10   | 0.20 | 0.30      | 0.30         |
| rule_based | 0.10   | 0.00 | 0.30      | 0.10         |

Leave-one-member-out ablation (`ablation_novelty.csv`; macro-F1 over the informative labels
PII/financial/confidential, benign shown per-label for transparency only):

| configuration    | macro-F1 | benign_f1 | PII_f1 | financial_f1 | confidential_f1 |
|------------------|----------|-----------|--------|--------------|-----------------|
| full_ensemble    | 0.974    | 0.927     | 0.991  | 0.954        | 0.979           |
| minus_bert       | 0.973    | 0.925     | 0.990  | 0.951        | 0.979           |
| minus_roberta    | 0.959    | 0.765     | 0.990  | 0.943        | 0.944           |
| minus_bilstm     | 0.972    | 0.929     | 0.991  | 0.948        | 0.977           |
| minus_rule_based | 0.970    | 0.933     | 0.991  | 0.939        | 0.979           |
| solo_roberta     | 0.972    | 0.928     | 0.990  | 0.948        | 0.977           |

Removing RoBERTa costs ~1.5 macro-F1 points (0.974 → 0.959; benign 0.927 → 0.765,
confidential 0.979 → 0.944, financial 0.954 → 0.943), while removing **any other** member
barely moves it (all ≥ 0.970). PII stays ~0.990 under *every* single removal, so no member
is load-bearing there. **Read honestly:**

- The learned weights *look* like specialization (financial split across three members;
  RoBERTa heavy on benign/PII/confidential), but the ablation shows the ensemble's real
  reliance is not cleanly category-differential — it leans on RoBERTa almost everywhere.
- The one category with a spread weight (financial) shows only mild reliance and **no
  collapse**: `minus_bilstm` financial = 0.948, essentially tied with `solo_roberta` =
  0.948. (An earlier "financial collapses when Bi-LSTM is dropped" artifact from a prior
  weight fit — which gave RoBERTa 0.0 on financial — is gone in this re-run, where RoBERTa
  carries a 0.30 financial weight.)
- So RQ2 is a **qualified no**: reliance-under-fixed-weights concentrated on RoBERTa, which
  is also why the RQ1 gain over RoBERTa-alone is so small.

**DLP operating-point** (`operating_point.csv`). Headline macro-F1 uses F1-optimal
thresholds, but DLP runs at high recall (a missed sensitive item costs far more than a
false alarm). At **recall ≥ 0.99**, the ensemble has the best per-category precision on all
three *informative* categories — PII 0.993, financial 0.621, confidential 0.964 — each ≥ the
best single model; on benign, BERT (0.489) beats the ensemble (0.427). *Caveats:* (a) the
thresholds meeting the recall floor are selected on the TEST split itself, so these are
optimistic upper bounds, not deployable estimates; (b) the edge is **not**
significance-tested; (c) the financial win sits at low precision (~0.62) where estimates
are noisy. Reported as a supporting qualifier to the (already small) RQ1 gain, not an
independent claim.

**Efficiency / deployment trade-off** (`efficiency.json`). The ensemble runs every member,
so its cost is the sum of all four (~241M parameters, dominated by the two transformers)
versus ~125M for RoBERTa alone — roughly 2× the transformer cost for a macro-F1 delta of
just +0.0026. Statistically significant, but small enough that the extra compute is hard to
justify for deployment.

## Honest caveats and threats to validity

- **Significance ≠ operational significance.** At N≈8k the paired bootstrap resolves
  tiny-but-real differences, so the RQ1 "win" over the transformers is reliable in
  direction but small in magnitude. The delta and its CI are always reported next to the
  p-value for exactly this reason.
- **RQ2 reliance is under fixed learned weights.** The ablation measures reliance given the
  DEV-fit per-category weights, not intrinsic model complementarity. It shows the ensemble
  leans on RoBERTa almost everywhere; it does not establish that the categories decompose
  cleanly across members.
- **Operating-point thresholds are chosen on the test split**, so the high-recall precision
  numbers are optimistic and un-tested for significance.
- **External validity.** All results are on ai4privacy (synthetic, sentence-level PII, with
  labels derived deterministically from the entity mask), not on real email. Absolute
  scores (~0.97) reflect an easy synthetic task; email DLP is the motivating application
  only. An email domain-shift probe is the largest genuine future novelty but is descoped
  in `CLAUDE.md` (no email corpus built).

## Design choices, justified against the prior art

- **Late fusion over early fusion.** Members are heterogeneous (transformers, Bi-LSTM,
  rule-based); there is no shared feature space, so probability-level fusion is natural.
- **Weighted pool over stacking.** Stacking is more expressive but needs more held-out data
  than our small dev split affords (comment 016); named as the stronger alternative we
  forgo for data-size reasons, not one we claim to beat.
- **Per-category weights over shared weights.** Per-category weighting is the natural
  instrument for *measuring* category-differential reliance (RQ2); it is standard as a
  mechanism, and here it doubles as the measurement tool that revealed the reliance is
  concentrated on RoBERTa rather than split by category.

## One-line summary

The fusion mechanism is standard per-class weighted soft voting (not novel). SentinelMail's
contribution is a rigorously significance-gated, multi-comparison project-level finding — on
ai4privacy sensitive-sentence classification, a per-category ensemble of heterogeneous NLP
models **beats** a single fine-tuned RoBERTa (RQ1: delta = +0.0026, 95% CI [+0.0006,
+0.0048], p = 0.006) and beats every member under Holm correction, but the gain over the
transformers is small; the ensemble's reliance is concentrated on RoBERTa rather than
cleanly per-category (RQ2: qualified no), which is why the RQ1 gain over RoBERTa-alone is so
small, with a modest un-tested precision edge at the DLP high-recall operating point — plus
a reusable, honesty-first evaluation harness. Demonstrated end-to-end in
`notebooks/07_novelty_analysis.ipynb`.
