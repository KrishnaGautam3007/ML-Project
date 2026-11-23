"""
Phase 3 (part 3) - Convert every cleaned function into a graph and save to disk.

We only want to build these graphs ONCE (parsing 30k functions takes a little
while), so we cache them as .pt files in data/graphs/. Phase 4 and Phase 5 then
just load the cached graphs.

Run:
    python src/build_graphs.py                 # all splits
    python src/build_graphs.py --limit 200     # quick test (first 200 rows)
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import torch
from tqdm import tqdm

from graph_builder import build_graph

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graphs")


def build_split(split: str, limit: int = 0):
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
    if limit:
        df = df.head(limit)

    graphs = []
    n_nodes = 0
    n_edges = 0  # undirected count
    failures = 0

    start = time.time()
    for code, label in tqdm(zip(df["code"], df["label"]), total=len(df), desc=split):
        try:
            g = build_graph(code, label=int(label))
            if g.num_nodes == 0:
                failures += 1
                continue
            graphs.append(g)
            n_nodes += g.num_nodes
            n_edges += g.edge_index.size(1) // 2
        except Exception:
            failures += 1

    elapsed = time.time() - start
    os.makedirs(GRAPH_DIR, exist_ok=True)
    out_path = os.path.join(GRAPH_DIR, f"{split}.pt")
    torch.save(graphs, out_path)

    k = len(graphs)
    print(f"  {split:<6}: {k:,} graphs | avg nodes {n_nodes/max(k,1):5.1f} | "
          f"avg undirected edges {n_edges/max(k,1):5.1f} | "
          f"failures {failures} | {elapsed:.1f}s")
    print(f"           saved -> {out_path}")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N rows per split (for testing)")
    args = ap.parse_args()

    print("Building graphs (AST + next-statement + variable-use edges)...")
    for split in ["train", "val", "test"]:
        build_split(split, limit=args.limit)
    print("\nPhase 3 done.")


if __name__ == "__main__":
    main()
