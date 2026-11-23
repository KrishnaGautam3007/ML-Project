"""
Stretch goal - train the Hybrid GraphSAGE WITH a second CWE-type head.

Two stages:
  Stage 1 (joint): train the whole model on BOTH tasks (risk + CWE type), with
          early stopping on the risk task's validation PR-AUC - same as train.py.
          Problem we observed: the risk task peaks after ~1 epoch, so training
          stops before the CWE head has learned anything.
  Stage 2 (head only): freeze the trained GNN, compute every function's graph
          embedding once, and train ONLY the small CWE head on those embeddings,
          with its own early stopping on validation macro-F1. The risk prediction
          is not touched by this stage.

We compare the CWE head with two baselines on vulnerable TEST functions:
  - majority class (always guess the most common training class)
  - TF-IDF + Logistic Regression on the code text ("code as plain text")

Prerequisite (once):   python src/cwe.py
Run:                   python src/train_cwe.py
Saves:                 models/hybrid_gnn_cwe.pt
"""

from __future__ import annotations

import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

from cwe import load_classes, cwe_index, IGNORE
from features import load_type_vocab
from gnn_data import load_graphs, prepare_graphs, make_loader
from train import set_seed, train_variant, save_main_model, DEVICE, PROJECT_ROOT
from evaluate import print_table

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def macro_f1(y_true, y_pred, C):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(C)), zero_division=0)


@torch.no_grad()
def cwe_embeddings(model, graphs):
    """Graph embeddings + CWE labels for the labelled (vulnerable) functions only."""
    model.eval()
    embs, ys = [], []
    for batch in make_loader(graphs, 128):
        batch = batch.to(DEVICE)
        emb, _ = model.embed(batch)
        embs.append(emb)
        ys.append(batch.cwe_y)
    E, Y = torch.cat(embs), torch.cat(ys)
    keep = Y != IGNORE
    return E[keep], Y[keep]


def train_head_only(model, Etr, Ytr, Eva, Yva, weights, C, epochs=300, patience=30):
    """Stage 2: train just model.cwe_head on frozen embeddings (full batch)."""
    head = model.cwe_head
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        loss = crit(head(Etr), Ytr)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            f1 = macro_f1(Yva.cpu().numpy(), head(Eva).argmax(1).cpu().numpy(), C)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    print(f"  stage 2: stopped at epoch {epoch}, best val macro-F1 {best:.3f}")


def tfidf_cwe_baseline(classes):
    """CWE type from code text only: TF-IDF + Logistic Regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def labelled(split):
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
        y = np.array([cwe_index(c, l, classes) for c, l in zip(df["cwe"], df["label"])])
        keep = y != IGNORE
        return df["code"][keep].tolist(), y[keep]

    Xtr, ytr = labelled("train")
    Xte, yte = labelled("test")
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2),
                          max_features=20000, min_df=2)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(Xtr), ytr)
    return yte, clf.predict(vec.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--cwe-weight", type=float, default=0.5,
                    help="how much the CWE loss counts relative to the risk loss")
    args = ap.parse_args()

    set_seed(42)
    classes = load_classes()
    if classes is None:
        raise SystemExit("Run `python src/cwe.py` first to create the CWE labels.")
    C = len(classes)
    vocab = load_type_vocab()

    tr = prepare_graphs(load_graphs("train", args.max_train), vocab)
    va = prepare_graphs(load_graphs("val"), vocab)
    te = prepare_graphs(load_graphs("test"), vocab)

    # class weights = inverse frequency, so the head can't just always say "Other"
    train_labels = [int(g.cwe_y) for g in tr if int(g.cwe_y) != IGNORE]
    counts = Counter(train_labels)
    weights = torch.tensor([len(train_labels) / (C * max(counts.get(i, 0), 1))
                            for i in range(C)], dtype=torch.float32)
    print("CWE classes:", classes)
    print("CWE class weights:", [round(float(w), 2) for w in weights])

    # ---- stage 1: joint training (risk + CWE) ----
    result, art = train_variant(
        "Hybrid GraphSAGE + CWE head", tr, va, te,
        conv_type="sage", use_metrics=True, epochs=args.epochs,
        patience=args.patience, num_cwe=C, cwe_weight=args.cwe_weight,
        cwe_class_weights=weights)
    model = art["model"]

    Etr, Ytr = cwe_embeddings(model, tr)
    Eva, Yva = cwe_embeddings(model, va)
    Ete, Yte = cwe_embeddings(model, te)
    y_true = Yte.cpu().numpy()
    with torch.no_grad():
        model.cwe_head.eval()
        joint_pred = model.cwe_head(Ete).argmax(1).cpu().numpy()

    # ---- stage 2: train only the CWE head on frozen embeddings ----
    train_head_only(model, Etr, Ytr, Eva, Yva, weights, C)
    with torch.no_grad():
        model.cwe_head.eval()
        head_pred = model.cwe_head(Ete).argmax(1).cpu().numpy()

    # ---- baselines ----
    majority = counts.most_common(1)[0][0]
    base_pred = np.full_like(y_true, majority)
    tf_true, tf_pred = tfidf_cwe_baseline(classes)

    print("\n================ CWE-type prediction (TEST, vulnerable functions) ================")
    print(f"labelled test functions: {len(y_true)} | classes: {classes}")
    print(f"{'Method':<40}{'Accuracy':>10}{'Macro-F1':>10}")
    print("-" * 60)
    rows = [
        (f"Majority class (always {classes[majority]})", y_true, base_pred),
        ("TF-IDF + LogReg (code as text)", tf_true, tf_pred),
        ("GNN CWE head, stage 1 (joint)", y_true, joint_pred),
        ("GNN CWE head, stage 2 (head-only)", y_true, head_pred),
    ]
    for name, yt, yp in rows:
        print(f"{name:<40}{accuracy_score(yt, yp):>10.3f}{macro_f1(yt, yp, C):>10.3f}")

    print("\nPer-class report (stage 2 head):")
    print(classification_report(y_true, head_pred, labels=list(range(C)),
                                target_names=classes, zero_division=0))

    print("=============== Main risk task of this model (TEST) ===============")
    print_table([result])
    print("(Single-task main model in README: PR-AUC 0.208, P@10% 0.203. One run each.)")

    save_main_model(art, filename="hybrid_gnn_cwe.pt", extra={"cwe_classes": classes})
    print("\nSaved -> models/hybrid_gnn_cwe.pt")


if __name__ == "__main__":
    main()
