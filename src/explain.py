"""
Phase 6 - Line-level explanations.

The model outputs ONE risk score per function. To point a reviewer at the right
place, we score every NODE, add those scores up per source LINE, and return the
top-3 riskiest lines with a short plain-English reason.

Node importance = attention weight  x  gradient-saliency magnitude.

  - ATTENTION weight (from the pooling layer): "how much did the model lean on
    this node when it summarised the whole function?" A value the model learned.
  - GRADIENT SALIENCY: "if I nudge this node's features a tiny bit, how much does
    the risk score move?" We get it from one backward pass: the size of the
    gradient of the score w.r.t. each node's feature vector. Big gradient = the
    score is sensitive to this node.

Multiplying them keeps only nodes that are BOTH highlighted by attention AND
actually move the prediction - a simple, standard way to combine the two.

We also do a Big-Vul-only evaluation: for truly vulnerable test functions, how
often is a line the security fix touched inside our top-3? ("Top-3 line hit rate".)
"""

from __future__ import annotations

import os
import re
import difflib

import numpy as np
import pandas as pd
import torch

from features import (node_features, function_metrics, normalize_metrics,
                      load_type_vocab, load_metric_stats, NODE_FEAT_DIM, METRIC_DIM)
from graph_builder import build_graph
from model import HybridGNN

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ---------------------------------------------------------------------------
# load the trained model + its calibration
# ---------------------------------------------------------------------------

def load_model():
    # map_location="cpu": the model was trained on GPU, but the backend scans on
    # CPU, and this also lets it run on a laptop with no GPU at all.
    ckpt = torch.load(os.path.join(MODELS_DIR, "hybrid_gnn.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = HybridGNN(cfg["in_dim"], cfg["metric_dim"], hidden=cfg["hidden"],
                      conv_type=cfg["conv_type"], use_metrics=cfg["use_metrics"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    vocab = load_type_vocab()
    stats = load_metric_stats()
    return model, vocab, stats, ckpt["temperature"], ckpt["threshold"]


def band(p: float) -> str:
    return "High" if p > 0.7 else ("Medium" if p >= 0.4 else "Low")


# ---------------------------------------------------------------------------
# build one graph ready for the model (features + metrics + batch)
# ---------------------------------------------------------------------------

def _prepare(code: str, vocab, stats):
    g = build_graph(code)                       # has node_types, node_tokens, line
    g.x = node_features(g.node_types, g.node_tokens, vocab)
    raw = function_metrics(code, g.node_types)
    g.metrics = torch.tensor(normalize_metrics(raw, stats), dtype=torch.float32).unsqueeze(0)
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)   # single-graph batch
    return g


# ---------------------------------------------------------------------------
# per-node importance = attention x gradient saliency
# ---------------------------------------------------------------------------

def node_importance(model, g):
    g.x = g.x.clone().requires_grad_(True)
    logit, alpha = model(g, return_attention=True)
    model.zero_grad()
    if g.x.grad is not None:
        g.x.grad.zero_()
    logit.backward()

    saliency = g.x.grad.abs().sum(dim=1)        # [N] how sensitive the score is
    importance = (alpha.detach() * saliency.detach()).cpu().numpy()
    return importance, float(logit.detach())


# ---------------------------------------------------------------------------
# template reasons
# ---------------------------------------------------------------------------

def reason_for_line(line_text: str, full_code: str) -> str:
    t = line_text
    if re.search(r"\b(strcpy|strcat)\s*\(", t):
        return "strcpy/strcat into a buffer with no length check"
    if re.search(r"\bmemcpy\s*\(", t):
        return "memcpy with a size not checked against the buffer"
    # pointer used after free(): the function frees something and this line
    # dereferences a pointer (-> or *p or p[..]) without being the free() call.
    if "free(" in full_code and "free(" not in t and re.search(r"(->|\*\w|\w\[)", t):
        return "pointer used after free() in this function"
    if re.search(r"\b(malloc|calloc|realloc)\s*\(", t):
        return "malloc result used without NULL check"
    return GENERIC_REASON


GENERIC_REASON = "unusual structure compared to safe code"


# ---------------------------------------------------------------------------
# the main entry point: explain one function
# ---------------------------------------------------------------------------

def explain_function(code: str, model, vocab, stats, temperature: float,
                     threshold: float, top_k: int = 3,
                     single_line_only: bool = True, body_only: bool = True,
                     pattern_first: bool = True) -> dict:
    """Score a function and return its top-k risky lines (index 0 = most likely).

    single_line_only=True: only nodes that sit on ONE line vote for that line.
    Big multi-line "container" nodes (the whole function, a { } body, a
    multi-line if) describe a block, not a line; if they voted, all their
    importance would land on their first line - usually the function
    signature, which then wrongly looks like "the vulnerable line".
    """
    g = _prepare(code, vocab, stats)
    importance, logit = node_importance(model, g)

    prob = float(torch.sigmoid(torch.tensor(logit / temperature)))

    lines = g.line.cpu().numpy()
    end_lines = g.end_line.cpu().numpy()

    def per_line_scores(skip_multiline: bool) -> dict[int, float]:
        scores: dict[int, float] = {}
        for imp, ln, end in zip(importance, lines, end_lines):
            if skip_multiline and ln != end:
                continue
            scores[int(ln)] = scores.get(int(ln), 0.0) + float(imp)
        return scores

    # sum node importance per 1-based line number
    per_line = per_line_scores(skip_multiline=single_line_only)
    if not per_line:                      # e.g. a one-line function
        per_line = per_line_scores(skip_multiline=False)

    # body_only=True: rank only lines INSIDE the function body. The signature
    # (return type, name, parameters) is a declaration, not where a bug happens,
    # but it is crowded with small nodes and would otherwise often win.
    # The body is the first `compound_statement` ({ ... }) in the function.
    if body_only and "compound_statement" in g.node_types:
        brace_line = int(lines[g.node_types.index("compound_statement")])
        body = {ln: s for ln, s in per_line.items() if ln > brace_line}
        if body:                          # keep everything for one-line functions
            per_line = body

    code_lines = code.split("\n")
    ranked = sorted(per_line.items(), key=lambda kv: kv[1], reverse=True)

    top_lines = []
    for ln, score in ranked[:top_k]:
        text = code_lines[ln - 1] if 1 <= ln <= len(code_lines) else ""
        top_lines.append({
            "line": ln,
            "score": round(score, 4),
            "text": text.strip(),
            "reason": reason_for_line(text, code),
        })

    # pattern_first=True: inside the model's top-k, move lines that match a known
    # dangerous pattern (i.e. got a specific reason) ahead of the rest. Python's
    # sort is stable, so the model's order is kept within each group. This only
    # reorders the top-k, so the Top-k hit rate cannot change - only Top-1.
    if pattern_first:
        top_lines.sort(key=lambda t: t["reason"] == GENERIC_REASON)

    return {
        "risk_score": round(prob, 3),
        "band": band(prob),
        "is_risky": prob >= threshold,
        "top_lines": top_lines,
    }


# ---------------------------------------------------------------------------
# ground-truth fixed lines (diff func_before vs func_after)
# ---------------------------------------------------------------------------

def changed_lines(before: str, after: str) -> set[int]:
    """1-based line numbers in `before` that the fix deleted or replaced."""
    b = before.split("\n")
    a = after.split("\n")
    sm = difflib.SequenceMatcher(a=b, b=a, autojunk=False)
    changed = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                changed.add(i + 1)       # 1-based
    return changed


# ---------------------------------------------------------------------------
# line-level evaluation on the test set
# ---------------------------------------------------------------------------

LINE_METHODS = {
    # name                              (single_line_only, body_only, pattern_first)
    "all nodes (original)":             (False, False, False),
    "single-line nodes":                (True, False, False),
    "single-line + body only":          (True, True, False),
    "single-line + body + pattern 1st": (True, True, True),
}


def evaluate_line_hit_rate(model, vocab, stats, temperature, threshold, top_k=3,
                           split="test"):
    """Top-1 and Top-k line hit rates for each line-scoring method + random.

    Choose a method on split="val", then report it on split="test", so the
    test numbers are not tuned on the test set.
    """
    from math import comb

    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
    vuln = df[df["label"] == 1]

    methods = LINE_METHODS
    hits = {m: 0 for m in methods}
    hits1 = {m: 0 for m in methods}
    rand_k, rand_1 = 0.0, 0.0
    evaluated = 0
    for code, after in zip(vuln["code"], vuln["func_after"]):
        gt = changed_lines(code, after)
        if not gt:
            continue                     # no usable ground truth, skip
        evaluated += 1
        n = code.count("\n") + 1
        k = len(gt)
        # expected hit rate if we picked lines at random
        rand_k += 1.0 if n <= top_k else 1 - comb(n - k, top_k) / comb(n, top_k)
        rand_1 += k / n
        for name, (single, body, pattern) in methods.items():
            exp = explain_function(code, model, vocab, stats, temperature,
                                   threshold, top_k, single_line_only=single,
                                   body_only=body, pattern_first=pattern)
            ranked = [t["line"] for t in exp["top_lines"]]
            if set(ranked) & gt:
                hits[name] += 1
            if ranked and ranked[0] in gt:
                hits1[name] += 1

    print(f"Line-level evaluation on {evaluated} vulnerable {split.upper()} functions")
    print("(hit = a line the real fix changed is among our top lines)\n")
    print(f"{'Method':<36}{'Top-1':>8}{'Top-' + str(top_k):>8}")
    print("-" * 52)
    print(f"{'random lines (expected)':<36}{rand_1 / evaluated:>8.3f}{rand_k / evaluated:>8.3f}")
    for name in methods:
        print(f"{name:<36}{hits1[name] / evaluated:>8.3f}{hits[name] / evaluated:>8.3f}")
    return {m: (hits1[m] / evaluated, hits[m] / evaluated) for m in methods}


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def _print_explanation(code: str, exp: dict):
    print(f"risk_score={exp['risk_score']}  band={exp['band']}  "
          f"risky={exp['is_risky']}")
    print("top lines:")
    for t in exp["top_lines"]:
        print(f"  L{t['line']:<3} (score {t['score']}) {t['reason']}")
        print(f"        | {t['text']}")


if __name__ == "__main__":
    model, vocab, stats, T, thr = load_model()

    demo = """void handle(char *input) {
    char buffer[64];
    strcpy(buffer, input);
    process(buffer);
}"""
    print("=== Demo explanation ===")
    exp = explain_function(demo, model, vocab, stats, T, thr)
    _print_explanation(demo, exp)

    # pick the line-scoring method on VALIDATION, then report it on TEST
    print("\n=== Line-level evaluation: choose method on VALIDATION ===")
    val = evaluate_line_hit_rate(model, vocab, stats, T, thr, split="val")
    best = max(val, key=lambda m: val[m])    # by Top-1, then Top-3
    print(f"\nChosen on validation: {best}")

    print("\n=== Line-level evaluation: report on TEST ===")
    evaluate_line_hit_rate(model, vocab, stats, T, thr, split="test")
