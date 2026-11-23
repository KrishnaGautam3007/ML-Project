"""
Phase 5 - loading cached graphs for training.

The graphs we saved in Phase 3/4 carry the raw node info (type, token, line) and
the normalised metric vector, but NOT the big 166-dim node feature matrix (we
left that out to keep the files small). Here we build that matrix once per graph
in memory, then drop the raw strings to save RAM.

If your laptop is tight on memory, pass a smaller --max-train in train.py.
"""

from __future__ import annotations

import os
import torch
from torch_geometric.loader import DataLoader

from features import node_features, load_type_vocab, NODE_FEAT_DIM, METRIC_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graphs")


def load_graphs(split: str, limit: int = 0):
    graphs = torch.load(os.path.join(GRAPH_DIR, f"{split}.pt"), weights_only=False)
    if limit and limit < len(graphs):
        # the graphs are grouped by project, so a plain [:limit] slice would be
        # skewed (one project's style, often all-safe). Sample randomly instead
        # so the capped set keeps a realistic class balance.
        import random
        rng = random.Random(42)
        graphs = rng.sample(graphs, limit)
    return graphs


def prepare_graphs(graphs, vocab, ast_only: bool = False):
    """Build node features in-place and (optionally) keep only AST edges.

    Returns the same list, now with g.x set and the raw string lists removed.
    """
    for g in graphs:
        # keep only AST edges (edge_type == 0) for the ablation
        if ast_only:
            mask = g.edge_type == 0
            g.edge_index = g.edge_index[:, mask]
            g.edge_type = g.edge_type[mask]

        # build the [num_nodes, 166] feature matrix once
        if getattr(g, "x", None) is None:
            g.x = node_features(g.node_types, g.node_tokens, vocab)

        # free the big python string lists we no longer need for training
        for attr in ("node_types", "node_tokens", "lang", "line", "end_line"):
            if attr in g:
                del g[attr]
    return graphs


def make_loader(graphs, batch_size: int = 64, shuffle: bool = False):
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)
