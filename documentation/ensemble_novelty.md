# Ensemble Novelty: Positioning Against Prior Fusion Work

This note states precisely what SentinelMail's late-fusion ensemble does and does not
claim as a contribution, and situates it against well-known ensemble-fusion literature.
It exists to keep the Novelty framing honest: the *weighting mechanism* is standard; the
contribution is the *empirical per-category complementarity finding* (RQ2).

Scope note: this doc is narrowly about fusion-method novelty. Broader related-work
grounding for DLP and PII classification is tracked separately (see comment 015); this
doc cross-references but does not duplicate that.

## The mechanism (explicitly NOT claimed as novel)

SentinelMail combines member models by a per-label normalized weighted average of
their probabilities:

    score_L = ( Σ_m w_{m,L} · p_{m,L} ) / ( Σ_m w_{m,L} )

with a separate weight vector fit per sensitivity category (`benign`, `PII`,
`financial`, `confidential`), the weights chosen to maximize per-label F1 on the
selection/dev split. Mechanistically this is **standard per-class weighted soft
voting** (a.k.a. label-wise weighted averaging / per-class combination of classifiers).
We do **not** claim the fusion rule, the per-class weighting, or the weight-fitting
procedure as new. Each of the following prior fusion families already covers the
mechanism:

- **Weighted soft voting / linear opinion pooling.** Averaging class posteriors with
  fixed or tuned weights is a classical combiner (Kittler et al., "On Combining
  Classifiers", 1998). Tuning the weights to optimize a validation metric is routine.
- **Per-class / label-wise classifier combination.** Assigning a different combiner
  weight per class (so a member trusted for one class can be down-weighted for another)
  is long established in the multiple-classifier-systems literature (Kuncheva,
  *Combining Pattern Classifiers*, 2004). Our per-category weights are an instance of
  this, not a departure from it.
- **Stacking / stacked generalization** (Wolpert, 1992). A learned meta-classifier on
  top of member outputs is strictly more expressive than our linear pool. We deliberately
  do NOT stack (see "Design choices" below); so stacking is prior art we forgo, not
  something we extend.
- **Mixture-of-experts gating** (Jacobs et al., 1991). MoE learns an input-conditioned
  gate that routes among experts. Our weights are input-independent (fixed per category,
  not per example), so this is a simpler, non-gated special case — again, not a novelty
  over MoE.

If the only claim were "we tune the weight separately per category," that claim would be
subsumed by per-class classifier combination and would not carry the Novelty mark.

## What IS claimed as the contribution (RQ2)

The contribution is **empirical, not mechanistic**: a per-category complementarity
finding for sensitive-sentence classification on ai4privacy — *which model family
dominates which sensitivity category, and by how much* — made legible through the
learned per-category weights and the leave-one-member-out ablation.

Concretely, the ensemble here is used as an **instrument** to measure complementarity:

- The learned per-category weight vectors are read as an interpretable signal of which
  member each category relies on (e.g. whether the spaCy+regex rule member carries
  `financial` via structured patterns while transformer members carry contextual
  categories). This turns the standard weighted-average combiner into a diagnostic for
  model/category specialization.
- The leave-one-member-out ablation (`ensemble.md` → Ablation Study) quantifies each
  member's marginal contribution *per category*, which is the direct evidence for RQ2.

This framing is the defensible novelty: not a new fusion algorithm, but a characterization
of *where and how much* different model types complement each other across sensitivity
categories in a DLP-motivated setting, on a task (multi-label sensitive-sentence
classification) where such a per-category breakdown has not, to our knowledge, been
reported.

**Strength caveat (do not overstate):** this contribution only holds if the ensemble
re-evaluation actually shows complementarity — i.e. per-category weights that differ
meaningfully across members and an ablation where different members matter for different
categories. That evidence depends on the pending held-out-test re-eval and the fair
rule-based `confidential` member (comments 001 and 007). Until those reruns land, the
per-category complementarity claim is *proposed*, not *demonstrated*. If the re-eval shows
uniform weights / no category-dependent complementarity, this claim must be weakened
accordingly rather than asserted.

## Design choices, justified against the prior art

- **Late fusion over early fusion.** Members are heterogeneous (transformers, Bi-LSTM,
  rule-based); there is no shared feature space to concatenate, so probability-level
  (late) fusion is the natural combiner. (Enforced in `ensemble.md`.)
- **Weighted pool over stacking.** A stacking meta-classifier is more expressive but needs
  enough held-out data to fit without overfitting; our selection/dev split is small
  (see comment 016), so a low-capacity linear pool is the appropriate choice. We name
  stacking as the stronger alternative we deliberately forgo for data-size reasons —
  this is a scope decision, not a claim that stacking is inferior in general.
- **Per-category weights over shared weights.** Per-category weighting is what makes the
  complementarity finding measurable; it is standard as a mechanism (per-class combination
  above), and here it is used as the measurement instrument for RQ2.

## One-line summary

The fusion mechanism is standard per-class weighted soft voting (not claimed novel). The
contribution is the empirical per-category complementarity finding (RQ2), made
interpretable via the learned per-category weights and the per-category leave-one-out
ablation — contingent on the pending re-eval actually showing complementarity.
