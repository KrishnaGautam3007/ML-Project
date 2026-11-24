"""
Phase 5 (runner) - train the Hybrid GraphSAGE, calibrate it, and run the
ablations. Produces ONE results table: Phase-2 baselines + main model + ablations.

Run:
    python src/train.py                 # full run (train 24k, ~15-30 min CPU)
    python src/train.py --max-train 4000 --epochs 12   # faster / low-RAM

What happens for each GNN variant:
  1. Train with Adam (lr 1e-3) and BCE loss weighted by pos_weight = #safe/#vuln,
     so the rare vulnerable class is not ignored.
  2. Early-stop on validation PR-AUC (keep the best epoch).
  3. Temperature-scale on validation so the probabilities are honest.
  4. Pick the decision threshold that maximises F1 on validation.
  5. Report Precision/Recall/F1/PR-AUC/P@10% on the TEST set.

The main model (SAGEConv + metrics) is saved to models/hybrid_gnn.pt for the
backend and the explanations.
"""

from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                                   # src/
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baselines"))  # baselines/

import calibration
from features import load_type_vocab, NODE_FEAT_DIM, METRIC_DIM
from gnn_data import load_graphs, prepare_graphs, make_loader
from model import HybridGNN
from evaluate import evaluate, print_table

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# prediction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    logits, ys = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits.append(model(batch).cpu())
        ys.append(batch.y.cpu())
    return torch.cat(logits), torch.cat(ys).float()


def best_f1_threshold(probs: np.ndarray, y: np.ndarray):
    """Pick the decision threshold with the best F1 on validation.

    Candidates are the upper quantiles of the validation probabilities: after
    calibration most probabilities are small (the vulnerable rate is ~6%), so a
    fixed grid like 0.05, 0.10, ... would be far too coarse.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(probs, np.linspace(0.50, 0.995, 200))):
        f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)



# ---------------------------------------------------------------------------
# one full train/eval of a variant
# ---------------------------------------------------------------------------

def train_variant(name, train_g, val_g, test_g, *, conv_type="sage",
                  use_metrics=True, epochs=30, patience=6, batch_size=64, lr=1e-3,
                  num_cwe=0, cwe_weight=0.5, cwe_class_weights=None):
    """Train one variant. With num_cwe > 0 (stretch goal) the model also learns
    a CWE-type head: total loss = risk BCE + cwe_weight * CWE cross-entropy,
    where the cross-entropy ignores functions whose cwe_y is -1 (safe/untyped)."""
    print(f"\n--- Training: {name} ---")
    train_loader = make_loader(train_g, batch_size, shuffle=True)
    val_loader = make_loader(val_g, batch_size, shuffle=False)
    test_loader = make_loader(test_g, batch_size, shuffle=False)

    # class imbalance weight from TRAIN
    y_train = torch.cat([g.y for g in train_g]).float()
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
    print(f"  pos_weight (safe/vuln) = {pos_weight.item():.1f}")

    model = HybridGNN(NODE_FEAT_DIM, METRIC_DIM, conv_type=conv_type,
                      use_metrics=use_metrics, num_cwe=num_cwe).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    # ignore_index=-1 skips safe / untyped functions in the CWE loss
    cwe_crit = None
    if num_cwe > 0:
        w = cwe_class_weights.to(DEVICE) if cwe_class_weights is not None else None
        cwe_crit = nn.CrossEntropyLoss(weight=w, ignore_index=-1)

    best_prauc, best_state, bad = -1.0, None, 0
    history = []        # one row per epoch: learning curves for the report
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            if cwe_crit is None:
                logit = model(batch)
                loss = crit(logit, batch.y.float())
            else:
                logit, cwe_logits = model(batch, return_cwe=True)
                loss = crit(logit, batch.y.float())
                # only add the CWE loss if this batch has at least one labelled
                # function (a batch of only ignored labels would give NaN)
                if (batch.cwe_y >= 0).any():
                    loss = loss + cwe_weight * cwe_crit(cwe_logits, batch.cwe_y)
            loss.backward()
            opt.step()
            total += float(loss) * batch.num_graphs

        # validation PR-AUC
        val_logits, val_y = predict_logits(model, val_loader)
        val_prob = torch.sigmoid(val_logits).numpy()
        prauc = average_precision_score(val_y.numpy(), val_prob)
        dt = time.time() - t0
        print(f"  epoch {epoch:>2} | loss {total/len(train_g):.4f} "
              f"| val PR-AUC {prauc:.4f} | {dt:.1f}s")
        history.append({"epoch": epoch, "train_loss": total / len(train_g),
                        "val_ap": float(prauc), "seconds": dt})

        if prauc > best_prauc:
            best_prauc = prauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val improvement for {patience} epochs)")
                break

    # restore best epoch
    model.load_state_dict(best_state)

    # calibrate (Platt scaling) + pick the threshold, both on validation only
    val_logits, val_y = predict_logits(model, val_loader)
    calib = calibration.fit_platt(val_logits.numpy(), val_y.numpy())
    val_prob = calibration.apply(calib, val_logits.numpy())
    thr, val_f1 = best_f1_threshold(val_prob, val_y.numpy())
    print(f"  Platt a = {calib['a']:.3f}, b = {calib['b']:.3f} | "
          f"F1-optimal threshold = {thr:.3f} (val F1 {val_f1:.3f})")

    # evaluate on test with calibrated probs
    test_logits, test_y = predict_logits(model, test_loader)
    test_prob = calibration.apply(calib, test_logits.numpy())
    test_pred = (test_prob >= thr).astype(int)
    result = evaluate(name, test_y.numpy(), test_pred, test_prob)
    result["val_ap"] = float(best_prauc)
    result["best_epoch"] = int(max(history, key=lambda h: h["val_ap"])["epoch"])

    artifacts = {"model": model, "calibration": calib, "threshold": thr,
                 "conv_type": conv_type, "use_metrics": use_metrics,
                 "num_cwe": num_cwe, "history": history,
                 "test_prob": test_prob, "test_y": test_y.numpy()}
    return result, artifacts


def save_main_model(artifacts, filename: str = "hybrid_gnn.pt", extra: dict | None = None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    torch.save({
        "state_dict": artifacts["model"].state_dict(),
        "config": {"in_dim": NODE_FEAT_DIM, "metric_dim": METRIC_DIM,
                   "conv_type": artifacts["conv_type"],
                   "use_metrics": artifacts["use_metrics"], "hidden": 128,
                   "num_cwe": artifacts.get("num_cwe", 0)},
        "calibration": artifacts["calibration"],
        "threshold": artifacts["threshold"],
        **(extra or {}),
    }, path)
    print(f"\nSaved main model -> {path}")


# ---------------------------------------------------------------------------
# baselines (re-used from Phase 2 so the final table is self-contained)
# ---------------------------------------------------------------------------

def baseline_rows(skip=False):
    if skip:
        return []
    import pandas as pd
    from run_baselines import run_flawfinder, run_tfidf_lr, run_risky_api, PROCESSED_DIR
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    y = test_df["label"].to_numpy()
    codes = test_df["code"].tolist()
    rows = []
    print("Scoring baselines for the final table...")
    p, s = run_flawfinder(codes); rows.append(evaluate("Flawfinder (level>=2)", y, p, s))
    p, s = run_tfidf_lr(train_df, test_df); rows.append(evaluate("TF-IDF + LogReg", y, p, s))
    p, s = run_risky_api(codes); rows.append(evaluate("Risky-API rule", y, p, s))
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=0, help="cap #train graphs (low RAM)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    vocab = load_type_vocab()
    print(f"Device: {DEVICE} | node feat dim {NODE_FEAT_DIM}")

    def fresh(ast_only=False):
        """Load + prepare a fresh copy of all three splits (each run mutates graphs)."""
        tr = prepare_graphs(load_graphs("train", args.max_train), vocab, ast_only)
        va = prepare_graphs(load_graphs("val"), vocab, ast_only)
        te = prepare_graphs(load_graphs("test"), vocab, ast_only)
        return tr, va, te

    results = baseline_rows(skip=args.skip_baselines)

    # ---- main model: SAGEConv + metrics, all edges ----
    tr, va, te = fresh(ast_only=False)
    main_res, main_art = train_variant(
        "Hybrid GraphSAGE (main)", tr, va, te,
        conv_type="sage", use_metrics=True,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
    results.append(main_res)
    save_main_model(main_art)

    if not args.skip_ablations:
        # ablation 1: AST edges only
        tr, va, te = fresh(ast_only=True)
        r, _ = train_variant("Ablation: AST edges only", tr, va, te,
                             conv_type="sage", use_metrics=True,
                             epochs=args.epochs, patience=args.patience,
                             batch_size=args.batch_size)
        results.append(r)

        # ablation 2: no metrics (pure GNN)
        tr, va, te = fresh(ast_only=False)
        r, _ = train_variant("Ablation: no metrics", tr, va, te,
                             conv_type="sage", use_metrics=False,
                             epochs=args.epochs, patience=args.patience,
                             batch_size=args.batch_size)
        results.append(r)

        # ablation 3: GCNConv instead of SAGEConv
        tr, va, te = fresh(ast_only=False)
        r, _ = train_variant("Ablation: GCN (not SAGE)", tr, va, te,
                             conv_type="gcn", use_metrics=True,
                             epochs=args.epochs, patience=args.patience,
                             batch_size=args.batch_size)
        results.append(r)

    print("\n\n================ PHASE 5 RESULTS (TEST set) ================")
    print_table(results)


if __name__ == "__main__":
    main()
