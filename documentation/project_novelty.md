# Project Novelty: What SentinelMail Contributes

This note states, honestly and precisely, what SentinelMail contributes as a research
project — not just the ensemble fusion method, but the project-level finding, its
mechanistic explanation, and the evaluation methodology used to reach it. Every
quantitative claim here is produced by `notebooks/07_novelty_analysis.ipynb` from the
cached member probabilities, with artifacts under `evaluation/results/`
(`rq1_verdict.json`, `ablation_novelty.csv`, `complementarity.json`,
`operating_point.csv`, `efficiency.json`) and figures under `report/figures/`.

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
  (ensemble − best single), tuned thresholds applied identically to every system
  (notebook §1). **Answer: no** — the delta is statistically indistinguishable from zero.

- **RQ2 — Does the ensemble's reliance on its members vary across sensitivity categories
  (PII vs. financial vs. confidential)?** Measured by the learned per-category weights and
  the leave-one-member-out ablation (notebook §2). **Answer: yes, the fitted ensemble
  leans on different members per category — but this reliance is confounded by the
  weight-fitting procedure and does not amount to a net benefit** (see caveats below).

The honest, defensible novelty of the project is the *combination* of these answers plus
the machinery that produced them, described next.

## The contribution, in three parts

1. **An empirical finding, rigorously significance-gated.** On multi-label
   sensitive-sentence classification (ai4privacy, DLP-motivated), a per-category weighted
   ensemble of four heterogeneous NLP members (BERT, RoBERTa, Bi-LSTM+char-CNN,
   spaCy/Presidio+regex) does **not** beat a single fine-tuned RoBERTa on headline
   macro-F1 — the gain is zero within a 95% CI. A negative result established this
   carefully (paired bootstrap, not eyeballed deltas) is a legitimate contribution, and
   the more so because a per-category ensemble is exactly the natural thing a practitioner
   would reach for.

2. **A mechanistic explanation for *why*.** We do not just report "no gain"; we explain
   it. An **oracle-ceiling** analysis shows the best single model already sits ~1–1.7
   points below the maximum any per-example combiner could reach, and an **error-overlap**
   analysis shows the members (especially the two transformers) make largely the same
   mistakes. With that little headroom and that much error correlation, no combiner —
   ours or a better one — had room to help.

3. **A reusable, honesty-first evaluation methodology.** The project packages a small
   harness for judging ensemble claims without overclaiming: paired-bootstrap
   significance gate (`ensemble_vs_best_single`), oracle-ceiling headroom
   (`oracle_gap`), error-set correlation (`error_overlap_matrix`), a DLP-relevant
   operating-point view (`precision_at_recall` / `operating_point_table`), and an
   efficiency/cost accounting. Each lives in the `evaluation` package and is exercised by
   notebook 07. The methodological point — *confirm an ensemble claim with an ablation AND
   a significance test, and judge at the operating point the application will actually
   use, not at F1-optimal thresholds* — is itself a transferable takeaway.

## In plain terms (read this first)

**What we built.** Four models that spot sensitive text (PII, financial, confidential):
two large language models (BERT, RoBERTa), one smaller neural network (Bi-LSTM), and one
rule-based checker (spaCy/Presidio + regex). Then an **ensemble** that asks all four and
blends their answers, with the blend weighted *per category*.

**What we hoped.** Different models are good at different things, so a smart per-category
blend should beat any single model — and mapping "who's good at what" would be a finding
in itself.

**What we found.** Tested fairly on held-out data, **the ensemble did not beat the best
single model** (RoBERTa): 0.9703 vs 0.9706 headline macro-F1 — a hair *worse*, and the gap
is statistically indistinguishable from zero.

**Does the per-category story hold up?** Partly, and with an honest asterisk. The fitted
ensemble does lean on different members for different categories (drop Bi-LSTM and the
financial score falls; drop BERT and PII falls; drop RoBERTa and confidential falls). But
this is reliance *under the specific fitted weights*, not proof of deep complementarity —
and for financial it is actually a symptom of a bad weight fit: the fitting gave RoBERTa a
weight of **0** on financial even though RoBERTa alone is the *best* financial model, so
the ensemble ends up slightly *worse* on financial than RoBERTa by itself.

**Why doesn't the ensemble win?** Because the best single model is already near the
ceiling. We measured an "oracle" — the score you'd get if you could always pick whichever
model happens to be right per example. RoBERTa alone is only ~1–1.7 points below it in
every category. With that little room, even genuine complementarity can only *match*
RoBERTa, not pass it. The models also make largely the *same* mistakes, so there's little
diversity to exploit.

**Anywhere it helps?** One place, with caveats: at the high-recall operating point DLP
uses (recall ≥ 99%), the ensemble shows the best per-category precision. But those
thresholds were picked on the test set itself (optimistic) and the edge is not
significance-tested, so we report it as a hint, not a win.

**So what's the contribution?** A carefully-tested negative result (the ensemble doesn't
beat the best single model), a clear explanation of why (oracle ceiling + correlated
errors), and a reusable, honesty-first way to evaluate ensemble claims — plus the caution
that plausible-looking learned weights and F1-optimal thresholds can both mislead.

**Terms used below:** *Macro-F1* — headline score averaged over PII/financial/confidential.
*Held-out/test* — data the models never trained on. *Ablation* — remove a model and
re-measure. *Paired bootstrap* — resamples the same rows for both systems to test whether
a gap is real (CI excludes 0) or noise.

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

**RQ1 — no significant ensemble gain** (held-out test, tuned per-label thresholds applied
identically to every system; comment 017 / comment 010; `rq1_verdict.json`):

    Ensemble macro-F1 = 0.97025   vs   best single (RoBERTa) = 0.97058
    delta = -0.00032   95% CI [-0.00293, +0.00237]   paired-bootstrap p = 0.80
    VERDICT: NOT SIGNIFICANT — delta CI includes 0.

The ensemble is a hair *worse* than RoBERTa alone. (Any larger "gain" seen elsewhere is an
artifact of a 4-label macro *including* `benign` and/or a fixed 0.5 threshold; under the
comment-011 headline metric — informative labels only, tuned thresholds — there is no gain.)

**RQ2 — the fitted ensemble's reliance varies by category (with a weight-fitting caveat).**
Learned per-category weights:

| model      | benign | PII      | financial | confidential |
|------------|--------|----------|-----------|--------------|
| bert       | 0.70   | **0.80** | 0.20      | 0.00         |
| roberta    | 0.10   | 0.00     | **0.00**  | **0.51**     |
| bilstm     | 0.10   | 0.10     | **0.50**  | 0.39         |
| rule_based | 0.10   | 0.10     | 0.30      | 0.10         |

Leave-one-member-out ablation (`ablation_novelty.csv`):

| configuration    | macro-F1 | PII_f1 | financial_f1 | confidential_f1 |
|------------------|----------|--------|--------------|-----------------|
| full_ensemble    | 0.970    | 0.991  | 0.946        | 0.975           |
| minus_bert       | 0.960    | 0.969  | 0.936        | 0.975           |
| minus_roberta    | 0.963    | 0.991  | 0.946        | 0.953           |
| minus_bilstm     | 0.874    | 0.990  | **0.656**    | 0.975           |
| minus_rule_based | 0.964    | 0.990  | 0.927        | 0.977           |
| solo_roberta     | 0.971    | 0.989  | **0.946**    | 0.976           |

Removing a member hurts the category it is weighted for (Bi-LSTM→financial, BERT→PII,
RoBERTa→confidential), so the ensemble's *reliance* is category-differential. **But do not
overread this as clean complementarity:**

- The financial collapse under `minus_bilstm` is largely an artifact. The weight fit gave
  **RoBERTa 0.0 on financial**, even though `solo_roberta` financial F1 = 0.946 — the best
  single financial model, and equal to the full ensemble. So the ensemble carries financial
  on Bi-LSTM+BERT+rule_based and is, if anything, **marginally worse on financial than
  RoBERTa alone**. Dropping Bi-LSTM then renormalizes onto the weaker remaining members and
  tanks. This shows sensitivity to a quirky weight fit, not that RoBERTa needs Bi-LSTM.
- Read honestly, RQ2 is "the fitted ensemble depends on different members per category,"
  which is weaker than "the models intrinsically complement each other," and it does not
  produce a net gain over the best single model (RQ1).

**Why no RQ1 win — the oracle ceiling** (`complementarity.json`). Per-category oracle upper
bound (accuracy achievable if one could pick the correct member per example):

| label        | best_single_acc | oracle_acc | oracle_gap |
|--------------|-----------------|------------|------------|
| PII          | 0.982           | 0.994      | 0.012      |
| financial    | 0.980           | 0.993      | 0.013      |
| confidential | 0.978           | 0.995      | 0.017      |

A ≤1.7-point gap leaves even a perfect combiner almost no headroom. The error-overlap
(Jaccard) tables in the same artifact show the two transformers share a large fraction of
their errors (~0.41 on PII, ~0.48 on confidential), consistent with the small headroom.
*Caveat:* the oracle bound is computed in **accuracy** while the headline metric is F1, so
it is a headroom proxy, not an F1-exact ceiling — the qualitative conclusion (little room)
is robust, but the exact number should be read as accuracy-space.

**DLP operating-point** (`operating_point.csv`). Headline macro-F1 uses F1-optimal
thresholds, but DLP runs at high recall (a missed sensitive item costs far more than a
false alarm). At **recall ≥ 0.99**, the ensemble has the best per-category precision of any
system (benign 0.47, PII 0.993, financial 0.647, confidential 0.960 — each ≥ the best
single model). *Caveats:* (a) the thresholds meeting the recall floor are selected on the
TEST split itself, so these are optimistic upper bounds, not deployable estimates; (b) the
edge is **not** significance-tested; (c) the benign/financial wins sit at low precision
where estimates are noisy. Reported as a qualifier, not an RQ1 reversal.

**Efficiency / deployment trade-off** (`efficiency.json`). The ensemble runs every member,
so its cost is the sum of all four (~241M parameters, dominated by the two transformers)
versus ~125M for RoBERTa alone — roughly 2× the transformer cost for a macro-F1 delta
indistinguishable from zero. Not justified for deployment.

## Honest caveats and threats to validity

- **RQ2 is confounded by the weight fit.** The grid+Nelder-Mead fit zero-weighted RoBERTa
  on financial despite it being the best single financial model, making the ensemble
  slightly worse on financial than RoBERTa alone. The category-differential reliance is
  therefore partly an artifact of the fitting, not pure model complementarity. A refit
  that allows RoBERTa a financial weight (or `fit_weights_lbfgsb`) would test whether the
  finding survives — flagged as follow-up, not yet done.
- **Operating-point thresholds are chosen on the test split**, so the high-recall
  precision numbers are optimistic and un-tested for significance.
- **Oracle bound is accuracy-space, headline is F1-space** — a proxy, not an exact ceiling.
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
  mechanism, and here it doubled as the measurement tool that also exposed the RoBERTa
  zero-weight-on-financial artifact.

## One-line summary

The fusion mechanism is standard per-class weighted soft voting (not novel). SentinelMail's
contribution is a rigorously significance-gated project-level finding — on ai4privacy
sensitive-sentence classification, a per-category ensemble of heterogeneous NLP models does
**not** beat a single fine-tuned RoBERTa (RQ1: delta ≈ 0, CI includes 0, p = 0.80) because
RoBERTa already sits within ~1.7 points of the oracle ceiling; the ensemble's per-category
*reliance* varies (RQ2) but is confounded by a quirky weight fit and yields no net gain,
with only an un-tested precision edge at the DLP high-recall operating point — plus a
reusable, honesty-first evaluation harness. Demonstrated end-to-end in
`notebooks/07_novelty_analysis.ipynb`.
