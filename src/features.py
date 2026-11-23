"""
Phase 4 - Features.

Two kinds of features:

A) NODE features (one row per graph node). Each node becomes a vector of:
     - one-hot of its AST type, limited to the ~100 most common types in the
       training set, plus a single "other" bucket for everything rarer   (101 dims)
     - a hashed bag-of-tokens of its text, 64 dims, HashingVectorizer-style
       (we hash the token's sub-words into 64 slots; no vocabulary to store) (64 dims)
     - 1 binary flag: is this node a risky C API name (strcpy, memcpy, ...)?  (1 dim)
   => NODE_FEAT_DIM = 166

B) FUNCTION-level metrics (one short vector per whole function - the "hybrid"
   part that gets glued onto the GNN's graph embedding in Phase 5):
     1. lines of code
     2. cyclomatic complexity  (1 + #if/for/while/case + #&& + #|| + #?)
     3. pointer operations     (count of *, ->, &)
     4. risky API call count
     5. max nesting depth       (deepest { } nesting)
     6. number of parameters
   These 6 numbers live on very different scales, so we z-normalise them using
   the mean/std measured on the TRAINING set only (never peeking at val/test).

We save two small files in models/:
   - type_vocab.json   : the 100 node types (so inference uses the same columns)
   - metric_stats.json : the train mean/std for the 6 metrics
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ---- sizes ----
NUM_TOP_TYPES = 100
TYPE_DIM = NUM_TOP_TYPES + 1          # +1 for the "other" bucket
TOKEN_HASH_DIM = 64
NODE_FEAT_DIM = TYPE_DIM + TOKEN_HASH_DIM + 1   # 101 + 64 + 1 = 166
METRIC_DIM = 6

RISKY_APIS = {"strcpy", "strcat", "gets", "sprintf", "memcpy", "scanf", "malloc", "free"}
_RISKY_CALL_RE = re.compile(r"\b(" + "|".join(sorted(RISKY_APIS)) + r")\s*\(")
_KEYWORD_RE = re.compile(r"\b(if|for|while|case)\b")
_SUBTOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# ===========================================================================
# A) NODE features
# ===========================================================================

def build_type_vocab(all_node_types: list[str]) -> dict[str, int]:
    """Pick the NUM_TOP_TYPES most common node types and map each to a column."""
    counts = Counter(all_node_types)
    top = [t for t, _ in counts.most_common(NUM_TOP_TYPES)]
    return {t: i for i, t in enumerate(top)}


def save_type_vocab(vocab: dict[str, int]):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "type_vocab.json"), "w") as fh:
        json.dump(vocab, fh)


def load_type_vocab() -> dict[str, int]:
    with open(os.path.join(MODELS_DIR, "type_vocab.json")) as fh:
        return json.load(fh)


def _hash_token(token: str, dim: int = TOKEN_HASH_DIM) -> np.ndarray:
    """Hash a token's sub-words into `dim` slots (the 'hashing trick')."""
    vec = np.zeros(dim, dtype=np.float32)
    if not token:
        return vec
    subs = _SUBTOKEN_RE.findall(token) or [token]
    for s in subs:
        # md5 (not python hash) so the mapping is stable across runs/machines
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 1.0
    return vec


def node_features(node_types: list[str], node_tokens: list[str],
                  vocab: dict[str, int]) -> torch.Tensor:
    """Build the [num_nodes, NODE_FEAT_DIM] feature matrix for one graph."""
    n = len(node_types)
    x = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
    other_col = NUM_TOP_TYPES      # the "other" bucket index within the type block
    for i in range(n):
        # a) one-hot type
        col = vocab.get(node_types[i], other_col)
        x[i, col] = 1.0
        # b) hashed token (offset by TYPE_DIM)
        x[i, TYPE_DIM:TYPE_DIM + TOKEN_HASH_DIM] = _hash_token(node_tokens[i])
        # c) risky-API flag (last column)
        if node_tokens[i] in RISKY_APIS:
            x[i, -1] = 1.0
    return torch.from_numpy(x)


# ===========================================================================
# B) FUNCTION-level metrics
# ===========================================================================

def function_metrics(code: str, node_types: list[str]) -> np.ndarray:
    """Compute the 6 hand-crafted metrics for one function (raw, un-normalised)."""
    # 1. lines of code
    loc = code.count("\n") + 1
    # 2. cyclomatic complexity
    cyclo = (1 + len(_KEYWORD_RE.findall(code))
             + code.count("&&") + code.count("||") + code.count("?"))
    # 3. pointer operations
    pointer_ops = code.count("->") + code.count("*") + code.count("&")
    # 4. risky API call count
    risky = len(_RISKY_CALL_RE.findall(code))
    # 5. max nesting depth (by braces)
    depth = max_depth = 0
    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    # 6. number of parameters (count parameter_declaration nodes in the AST)
    n_params = node_types.count("parameter_declaration")

    return np.array([loc, cyclo, pointer_ops, risky, max_depth, n_params],
                    dtype=np.float32)


METRIC_NAMES = ["loc", "cyclomatic", "pointer_ops", "risky_calls", "max_nesting", "n_params"]


def fit_metric_stats(metrics: np.ndarray) -> dict:
    """metrics: [N, 6] raw. Return {'mean':[...], 'std':[...]} from train data."""
    mean = metrics.mean(axis=0)
    std = metrics.std(axis=0)
    std[std < 1e-6] = 1.0          # avoid divide-by-zero for a constant column
    return {"mean": mean.tolist(), "std": std.tolist()}


def save_metric_stats(stats: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "metric_stats.json"), "w") as fh:
        json.dump(stats, fh)


def load_metric_stats() -> dict:
    with open(os.path.join(MODELS_DIR, "metric_stats.json")) as fh:
        return json.load(fh)


def normalize_metrics(raw: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    return (raw - mean) / std
