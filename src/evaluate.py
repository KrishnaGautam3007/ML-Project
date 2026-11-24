"""
Shared evaluation metrics - used by the baselines (Phase 2) AND by the GNN
(Phase 5). Everything that predicts "is this function vulnerable?" is scored
with the SAME functions here, so comparisons are fair.

Each model gives us, for every test function:
  - y_true  : the real label (1 = vulnerable, 0 = safe)
  - y_pred  : the model's 0/1 decision
  - y_score : a continuous "riskiness" number (higher = riskier). Used for
              ranking-based metrics. For a pure rule, this can just be a hit
              count or severity level.

Metrics we report:
  - Precision / Recall / F1 for the VULNERABLE class.
  - PR-AUC  (area under the Precision-Recall curve).
  - Precision@Top-10%.

WHY PR-AUC INSTEAD OF ACCURACY?
  Our data is imbalanced: only ~5-6% of functions are vulnerable. A lazy model
  that predicts "safe" for everything gets ~94% accuracy while catching zero
  bugs - useless. PR-AUC looks only at how well we rank the *vulnerable* class
  across every possible threshold, so it cannot be fooled by the large safe
  majority. A random model scores PR-AUC ~= the vulnerable rate (~0.05), and 1.0
  is perfect.

WHAT IS PRECISION@TOP-10%?
  In real life a reviewer only has time to check a few functions. So we rank all
  test functions from riskiest to safest, take the top 10%, and ask: what
  fraction of THOSE are actually vulnerable? It answers "if the tool gives me a
  short worklist, how much of it is worth my time?".
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def precision_at_top_k_percent(y_true, y_score, k: float = 0.10) -> float:
    """Of the top k-fraction riskiest functions, what fraction are vulnerable?

    Ties: rule-based scores (Flawfinder levels, risky-call counts, line counts)
    give many functions the SAME score. If the top-k cut-off falls inside such
    a tied group, which members get in is arbitrary. Taking the first ones in
    file order would make the result depend on how the rows happen to be
    sorted. Instead we count the tied group at its average vulnerable rate,
    i.e. the expected value under random tie-breaking.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    top_n = max(1, int(round(n * k)))
    cutoff = np.sort(y_score)[::-1][top_n - 1]        # score of the last function let in
    above = y_score > cutoff
    tied = y_score == cutoff
    slots_left = top_n - int(above.sum())               # places filled from the tied group
    expected_hits = y_true[above].sum() + slots_left * y_true[tied].mean()
    return float(expected_hits / top_n)


def evaluate(name: str, y_true, y_pred, y_score) -> dict:
    """Compute all metrics for one model and return them in a dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "name": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # PR-AUC is undefined if the test set has only one class.
        "pr_auc": average_precision_score(y_true, y_score) if has_both_classes else float("nan"),
        "p_at_10": precision_at_top_k_percent(y_true, y_score, 0.10),
    }


def print_table(results: list[dict]):
    """Pretty-print a list of result dicts as one comparison table."""
    header = f"{'Model':<30}{'Prec':>7}{'Recall':>8}{'F1':>7}{'PR-AUC':>8}{'P@10%':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<30}"
            f"{r['precision']:>7.3f}"
            f"{r['recall']:>8.3f}"
            f"{r['f1']:>7.3f}"
            f"{r['pr_auc']:>8.3f}"
            f"{r['p_at_10']:>8.3f}"
        )
