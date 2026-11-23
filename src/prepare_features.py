"""
Phase 4 (runner) - fit the feature vocab + metric stats on the TRAINING set,
attach the normalised metric vector to every cached graph, and print:
  - the final feature shapes
  - a mean comparison of each metric for vulnerable vs safe functions.

Node features themselves are built on the fly in Phase 5 (cheap, and it keeps
the cached files small / laptop-friendly); here we only need the type vocab so
that train and inference use the same columns.

The cached graphs (data/graphs/*.pt) line up 1:1 with the parquet rows because
Phase 3 had zero failures - we assert that to be safe.

Run:
    python src/prepare_features.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graphs")


def load_split(split: str):
    graphs = torch.load(os.path.join(GRAPH_DIR, f"{split}.pt"), weights_only=False)
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
    assert len(graphs) == len(df), (
        f"{split}: {len(graphs)} graphs vs {len(df)} rows - alignment broken. "
        "Re-run src/build_graphs.py."
    )
    return graphs, df


def main():
    # ---- 1. build the node-type vocabulary from TRAIN graphs ----
    print("Loading train graphs...")
    train_graphs, train_df = load_split("train")

    all_types = []
    for g in train_graphs:
        all_types.extend(g.node_types)
    vocab = F.build_type_vocab(all_types)
    F.save_type_vocab(vocab)
    print(f"Type vocab: kept top {len(vocab)} node types + 1 'other' bucket.")
    print("  most common:", list(vocab.keys())[:8], "...")

    # ---- 2. compute raw metrics for every split ----
    raw_metrics = {}
    splits = {"train": (train_graphs, train_df)}
    for split in ["val", "test"]:
        splits[split] = load_split(split)

    for split, (graphs, df) in splits.items():
        m = np.stack([
            F.function_metrics(code, g.node_types)
            for code, g in zip(df["code"], graphs)
        ])
        raw_metrics[split] = m

    # ---- 3. fit normalisation stats on TRAIN only, save ----
    stats = F.fit_metric_stats(raw_metrics["train"])
    F.save_metric_stats(stats)
    print("\nSaved metric stats (train mean/std).")

    # ---- 4. attach normalised metrics to each graph and re-save ----
    for split, (graphs, df) in splits.items():
        norm = F.normalize_metrics(raw_metrics[split], stats)
        for g, row in zip(graphs, norm):
            g.metrics = torch.tensor(row, dtype=torch.float32).unsqueeze(0)  # [1,6]
        torch.save(graphs, os.path.join(GRAPH_DIR, f"{split}.pt"))
    print("Attached normalised metric vectors to all cached graphs.")

    # ---- 5. report feature shapes ----
    g0 = train_graphs[0]
    x0 = F.node_features(g0.node_types, g0.node_tokens, vocab)
    print("\n=== Feature shapes ===")
    print(f"  node feature matrix (example graph): {tuple(x0.shape)}  "
          f"(NODE_FEAT_DIM = {F.NODE_FEAT_DIM})")
    print(f"      = {F.TYPE_DIM} type one-hot + {F.TOKEN_HASH_DIM} hashed token "
          f"+ 1 risky flag")
    print(f"  metric vector per function: (1, {F.METRIC_DIM})")

    # ---- 6. vulnerable vs safe metric comparison (on train) ----
    y = train_df["label"].to_numpy()
    m = raw_metrics["train"]
    vuln = m[y == 1]
    safe = m[y == 0]
    print("\n=== Metric means: vulnerable vs safe (train, raw values) ===")
    print(f"{'metric':<14}{'vulnerable':>12}{'safe':>10}{'ratio v/s':>11}")
    print("-" * 47)
    for j, name in enumerate(F.METRIC_NAMES):
        v, s = vuln[:, j].mean(), safe[:, j].mean()
        ratio = v / s if s else float("nan")
        print(f"{name:<14}{v:>12.2f}{s:>10.2f}{ratio:>11.2f}")

    print("\nPhase 4 done.")


if __name__ == "__main__":
    main()
