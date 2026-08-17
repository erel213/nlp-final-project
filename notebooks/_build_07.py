"""Generator for notebooks/07_novelty_analysis.ipynb.

Kept in-repo so the notebook is reproducible from source. Run:
    .venv/bin/python notebooks/_build_07.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ---------------------------------------------------------------- Section 0
md("""# 07 — Novelty Analysis: the ensemble negative result and *why*

This notebook produces the evidence behind SentinelMail's novelty claim, which is an
**honest, significance-gated negative result** (see `documentation/project_novelty.md`):

> On ai4privacy sensitive-sentence classification, a per-category weighted ensemble of
> four heterogeneous NLP members does **not** significantly beat a single fine-tuned
> RoBERTa — *even though* its learned per-category weights look like clean model
> specialization.

It answers the two research questions and adds three supporting analyses that explain
and qualify the result:

1. **RQ1** — does the ensemble beat the best single model? (headline verdict)
2. **RQ2** — does model complementarity vary across sensitivity categories?
3. **Error-complementarity** — *why* fusion can't help (correlated errors / oracle gap).
4. **DLP operating-point** — does the ensemble help at high recall, the point DLP needs?
5. **Efficiency** — the ensemble's cost vs. a single model, for no significant gain.

Everything runs post-hoc on the member probabilities
(computed fresh via `ensemble.compute_probas` — no on-disk cache); nothing is
retrained. All evaluation logic comes from the
`evaluation` package (no direct sklearn/matplotlib in this notebook).""")

# ---------------------------------------------------------------- Setup
md("## Setup")
code("""import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("..").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
ENSEMBLE_DIR = REPO_ROOT / "ensemble"
FIG_DIR = REPO_ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42""")

code("""import torch

from ensemble import compute_probas
from evaluation import (
    LABEL_COLS, MACRO_LABELS,
    load_weights, fuse, run_ablation, ensemble_vs_best_single, compute_all_metrics,
    plot_ablation_table, plot_pr_curves, plot_error_overlap,
    # novelty helpers (evaluation/complementarity.py, evaluation/operating_point.py)
    per_model_correct, oracle_gap, error_overlap_matrix, unique_contribution,
    complementarity_summary, precision_at_recall, operating_point_table,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# DEV split (fits weights/thresholds) and TEST split (reported once). Same splits as
# notebook 06. compute_probas runs inference fresh (no on-disk cache), so these
# probabilities always reflect the current checkpoints.
dev_probas, dev_y, dev_texts = compute_probas(split="train_holdout", device=DEVICE)
eval_probas, eval_y, eval_texts = compute_probas(split="validation", device=DEVICE)

weights = load_weights(ENSEMBLE_DIR / "weights.json")
with open(ENSEMBLE_DIR / "thresholds.json") as f:
    thresholds = json.load(f)

MODELS = list(eval_probas)
print("models:", MODELS)
print("TEST N:", eval_y.shape[0], "| DEV N:", dev_y.shape[0])
print("tuned thresholds:", thresholds)""")

# ---------------------------------------------------------------- Section 1 RQ1
md("""## 1 — RQ1: does the ensemble beat the best single model?

Paired bootstrap on the headline macro-F1 delta (ensemble − best single), the *same*
resampled TEST rows for both systems, with the DEV-tuned per-label thresholds applied
identically to every configuration (apples-to-apples; comments 010/017). We claim a gain
only if the delta's 95% CI excludes 0.""")
code("""rq1 = ensemble_vs_best_single(eval_probas, weights, eval_y, threshold=thresholds)

print(f"Ensemble macro-F1     = {rq1['ensemble_macro_f1']:.5f}")
print(f"Best single ({rq1['best_single_model']}) = {rq1['best_single_macro_f1']:.5f}")
print(f"delta = {rq1['mean_delta']:+.5f}  "
      f"95% CI [{rq1['ci_lower']:+.5f}, {rq1['ci_upper']:+.5f}]  "
      f"bootstrap p = {rq1['p_value']:.4f}")
verdict = ("SIGNIFICANT — ensemble beats the best single model"
           if rq1['significant'] else
           "NOT SIGNIFICANT — delta CI includes 0; no ensemble gain over the best single model")
print("RQ1 VERDICT:", verdict)

with open(RESULTS_DIR / "rq1_verdict.json", "w") as f:
    json.dump(rq1, f, indent=2)
print("saved -> evaluation/results/rq1_verdict.json")""")

# ---------------------------------------------------------------- Section 2 RQ2
md("""## 2 — RQ2: does complementarity vary across categories?

The learned per-category weights *look* like specialization — each category leans on a
different member. But the leave-one-out ablation tests whether those differences are
**load-bearing**: if removing a member barely moves the per-category F1, its apparent
"specialization" buys nothing.""")
code("""weights_df = pd.DataFrame(weights).T[LABEL_COLS].round(3)  # rows=model, cols=label
print("Learned per-category weights (rows=model, cols=label):")
weights_df""")
code("""ablation_df = run_ablation(eval_probas, weights, eval_y, threshold=thresholds)
ablation_df.to_csv(RESULTS_DIR / "ablation_novelty.csv", index=False)
print("saved -> evaluation/results/ablation_novelty.csv")
plot_ablation_table(ablation_df)""")
md("""**Read:** compare `full_ensemble` against each `minus_{model}` row per category.
Removing a member *does* hurt the category it is weighted for (drop Bi-LSTM → `financial`
collapses; drop BERT → `PII` drops; drop RoBERTa → `confidential` drops), so the fitted
ensemble's reliance is genuinely category-differential (RQ2: yes).

**But read it honestly — a weight-fitting confound.** The financial collapse under
`minus_bilstm` is largely an artifact: the fit gave **RoBERTa weight 0.0 on financial**
even though `solo_roberta` is the *best* single financial model (F1 ≈ 0.946, equal to the
full ensemble). So the ensemble carries financial on the weaker members and is, if
anything, marginally *worse* on financial than RoBERTa alone; dropping Bi-LSTM then
renormalizes onto those weaker members and tanks. This shows sensitivity to a quirky
weight fit, not intrinsic complementarity — and it does not produce a net gain (RQ1).""")

# ---------------------------------------------------------------- Section 3 mechanism
md("""## 3 — Error-complementarity: *why* fusion can't help

An ensemble only improves on its best member when members make **different** mistakes.
We reduce each member's cached probabilities to per-example correctness (at the tuned
thresholds) and measure:

- **Oracle gap** — best-single accuracy vs. the oracle upper bound (recoverable if *any*
  member is correct). A small gap = little headroom for *any* combiner.
- **Error overlap (Jaccard)** — how correlated the members' error sets are per category.
- **Unique contribution** — examples only one member gets right (exclusive catches).""")
code("""correct = per_model_correct(eval_probas, eval_y, threshold=thresholds)

gap_df = oracle_gap(correct, eval_y)
print("Oracle gap per label (oracle_acc - best_single_acc):")
gap_df.round(4)""")
code("""uniq_df = unique_contribution(correct, eval_y)
print("Exclusive correct catches per (label, model):")
uniq_df.pivot(index="model", columns="label", values="exclusive_frac").round(4)[LABEL_COLS]""")
code("""# Per-category error-overlap (Jaccard of error sets); 1.0 = identical errors.
overlap_tables = {lab: error_overlap_matrix(correct, lab) for lab in MACRO_LABELS}
for lab in MACRO_LABELS:
    print(f"\\n=== error overlap (Jaccard) — {lab} ===")
    print(overlap_tables[lab].round(3))""")
code("""# Heatmap figure of error overlap per informative category (evaluation wrapper; no
# direct matplotlib in the notebook).
_, fig = plot_error_overlap(overlap_tables, return_fig=True)
fig.savefig(FIG_DIR / "error_overlap.png", dpi=150, bbox_inches="tight")
print("saved -> report/figures/error_overlap.png")""")
code("""# Persist the mechanism summary for the report.
summary = complementarity_summary(eval_probas, eval_y, threshold=thresholds)
with open(RESULTS_DIR / "complementarity.json", "w") as f:
    json.dump(summary, f, indent=2)
print("saved -> evaluation/results/complementarity.json")
print("max oracle gap over informative labels:",
      round(gap_df[gap_df.label.isin(MACRO_LABELS)].oracle_gap.max(), 4))""")

# ---------------------------------------------------------------- Section 4 operating pt
md("""## 4 — DLP operating point: precision at high recall

Macro-F1 at an F1-optimal threshold is the wrong lens for DLP, where a missed sensitive
item costs far more than a false alarm. We compare each single model and the fused
ensemble on **precision at recall ≥ 0.99** per category — the high-recall regime DLP
actually runs in. A NaN means the recall floor is unreachable (e.g. the rule-based
member's step-function scores).""")
code("""TARGET_RECALL = 0.99
op_df = operating_point_table(eval_probas, eval_y, weights=weights, target_recall=TARGET_RECALL)
op_df.to_csv(RESULTS_DIR / "operating_point.csv")
print(f"Precision at recall >= {TARGET_RECALL} (rows=model incl. fused ensemble):")
print("saved -> evaluation/results/operating_point.csv")
op_df.round(4)""")
code("""# PR curves for the fused ensemble (context for the operating-point table).
_, fused_proba = fuse(eval_probas, weights)
_, fig = plot_pr_curves(eval_y, fused_proba, return_fig=True)
fig.savefig(FIG_DIR / "pr_curves_novelty.png", dpi=150, bbox_inches="tight")
print("saved -> report/figures/pr_curves_novelty.png")""")
md("""**Read:** at recall ≥ 0.99 the `ensemble` row has the best per-category precision of
any system — the one place the ensemble shows an edge that headline macro-F1 (RQ1) hides.
**Caveats:** the thresholds meeting the recall floor are selected on this TEST split
itself (optimistic, not a deployable estimate), the edge is **not** significance-tested,
and the benign/financial wins sit at low precision where estimates are noisy. Report it as
a qualifier, not an RQ1 reversal.""")

# ---------------------------------------------------------------- Section 5 efficiency
md("""## 5 — Efficiency: cost vs. no gain

The ensemble must execute **every** member at inference, so its cost is the sum of all
four members' costs — dominated by the two transformers. We report each member's
parameter count (from its checkpoint); the rule-based member has zero trained
parameters. The takeaway: the ensemble roughly doubles the transformer cost of running
RoBERTa alone, for a macro-F1 delta that is statistically indistinguishable from zero
(Section 1).""")
code("""import torch

def _count_params(pt_path):
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        return None
    total = 0
    for v in state.values():
        if isinstance(v, torch.Tensor):
            total += v.numel()
    return total

ckpts = {
    "bert": REPO_ROOT / "models/bert/checkpoint/best_model.pt",
    "roberta": REPO_ROOT / "models/roberta/checkpoint/best_model.pt",
    "bilstm": REPO_ROOT / "models/bilstm/checkpoint/best_model.pt",
}
param_counts = {"rule_based": 0}
for name, path in ckpts.items():
    param_counts[name] = _count_params(path) if path.exists() else None

eff = {
    "param_counts": param_counts,
    "ensemble_total_params": sum(v for v in param_counts.values() if v),
    "note": ("ensemble runs all members; inference cost ~= sum of members, dominated by "
             "the two transformers. RQ1 gain over best single is not significant."),
}
with open(RESULTS_DIR / "efficiency.json", "w") as f:
    json.dump(eff, f, indent=2)
print("saved -> evaluation/results/efficiency.json")
pd.DataFrame({"params": param_counts}).assign(
    millions=lambda d: (d["params"] / 1e6).round(1))""")

# ---------------------------------------------------------------- Section 6 summary
md("""## 6 — Summary

Putting the five analyses together (all on the ai4privacy TEST split, headline macro-F1
over PII/financial/confidential):

- **RQ1:** the per-category ensemble does **not** significantly beat the best single
  model (RoBERTa) — delta ≈ 0, 95% CI includes 0.
- **RQ2:** the fitted ensemble's reliance *does* vary by category (leave-one-out hurts the
  weighted category), so the answer is yes — but it is confounded by the weight fit
  (RoBERTa zero-weighted on financial despite being the best single financial model), so
  it is reliance-under-fixed-weights, not clean intrinsic complementarity, and yields no
  net gain.
- **Mechanism:** members make highly correlated errors (large error-set Jaccard) and the
  oracle gap is small (≤1.7 pts, accuracy-space), so there is little headroom for *any*
  combiner — the direct reason fusion can't pull ahead.
- **DLP operating point:** the ensemble has the best per-category precision at
  recall ≥ 0.99 — a small edge the headline metric hides, but with thresholds picked on
  test and no significance test, so a qualifier only.
- **Efficiency:** the ensemble costs ~the sum of all members (~241M params, two
  transformers dominate) vs ~125M for RoBERTa alone, for that non-significant gain — not
  justified for deployment.

This is the project's honest contribution: a rigorously significance-gated finding (RQ1
negative; RQ2 category-differential reliance, confounded by the weight fit) plus its
mechanism, and a reusable, honesty-first evaluation harness — with the caution that
learned weights *and* F1-optimal thresholds can both mislead. See
`documentation/project_novelty.md` for the full write-up.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = "notebooks/07_novelty_analysis.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
