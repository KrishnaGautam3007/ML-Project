"""
Repeat the main model and the three ablations with several random seeds.

One training run is not enough to compare variants whose PR-AUC differs by
~0.02: the test set has only 193 vulnerable functions, and GNN training has
randomness (weight initialisation, batch order, dropout). This script trains
each variant with seeds 1-5 and records, for every run:
    - test PR-AUC (average precision), Precision@Top-10%, precision/recall/F1
    - best validation PR-AUC and the epoch it happened at
    - the per-epoch learning curve (train loss, validation PR-AUC)
    - the calibrated test probabilities (for later analysis)

It never overwrites the deployed model (models/hybrid_gnn.pt).

Run:  python src/run_seeds.py            (~20-30 min on a GPU)
Out:  results/seed_runs.json, results/seed_test_probs.npz
"""

from __future__ import annotations

import os
import json
import time
import argparse

import numpy as np

from features import load_type_vocab
from gnn_data import load_graphs, prepare_graphs
from train import set_seed, train_variant, PROJECT_ROOT

RESULTS = os.path.join(PROJECT_ROOT, "results")

VARIANTS = [
    # name,                       conv,  metrics, ast_only
    ("Hybrid GraphSAGE (main)", "sage", True, False),
    ("Ablation: no metrics", "sage", False, False),
    ("Ablation: GCN (not SAGE)", "gcn", True, False),
    ("Ablation: AST edges only", "sage", True, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    vocab = load_type_vocab()

    # training does not modify the graphs, so each edge configuration is
    # loaded and prepared only once and re-used for every seed
    data = {}
    for ast_only in (False, True):
        data[ast_only] = tuple(prepare_graphs(load_graphs(s), vocab, ast_only)
                               for s in ("train", "val", "test"))

    runs, probs = [], {}
    t_start = time.time()
    for seed in args.seeds:
        for name, conv, metrics, ast_only in VARIANTS:
            set_seed(seed)
            tr, va, te = data[ast_only]
            result, art = train_variant(f"{name} [seed {seed}]", tr, va, te,
                                        conv_type=conv, use_metrics=metrics)
            runs.append({
                "variant": name, "seed": seed,
                "test_ap": result["pr_auc"], "test_p_at_10": result["p_at_10"],
                "test_precision": result["precision"], "test_recall": result["recall"],
                "test_f1": result["f1"], "val_ap": result["val_ap"],
                "best_epoch": result["best_epoch"], "history": art["history"],
                "calibration": art["calibration"], "threshold": art["threshold"],
            })
            probs[f"{name}|{seed}"] = art["test_prob"]
            probs["y"] = art["test_y"]
            # save after every run, so nothing is lost if it is interrupted
            with open(os.path.join(RESULTS, "seed_runs.json"), "w") as fh:
                json.dump(runs, fh, indent=1)
            np.savez(os.path.join(RESULTS, "seed_test_probs.npz"), **probs)

    print(f"\nDone: {len(runs)} runs in {(time.time() - t_start) / 60:.1f} min")
    print(f"{'variant':<28}{'test AP mean±std':>20}{'P@10 mean±std':>18}{'val AP':>10}")
    for name, *_ in VARIANTS:
        r = [x for x in runs if x["variant"] == name]
        aps = np.array([x["test_ap"] for x in r])
        p10 = np.array([x["test_p_at_10"] for x in r])
        va = np.array([x["val_ap"] for x in r])
        print(f"{name:<28}{aps.mean():>12.3f} ± {aps.std(ddof=1):.3f}"
              f"{p10.mean():>11.3f} ± {p10.std(ddof=1):.3f}{va.mean():>10.3f}")


if __name__ == "__main__":
    main()
