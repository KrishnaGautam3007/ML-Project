"""
Stretch goal - CWE-type labels for the second output head.

The main model says HOW RISKY a function is. The CWE head guesses WHAT KIND of
weakness it most likely has (CWE = Common Weakness Enumeration, e.g. CWE-119 =
buffer errors). We predict the 5 most common CWEs in the TRAINING set plus one
"Other" class.

Which functions train the CWE head?
  - only VULNERABLE functions, and only those with a real CWE label.
  - safe functions are masked out: in Big-Vul every function touched by a fixing
    commit inherits that commit's CWE, so a safe function's CWE tag is meaningless.
  - vulnerable functions labelled "None" are masked out too (type unknown).
Masked functions get cwe_y = -1, and the loss simply ignores them.

Run once (after Phase 4):
    python src/cwe.py
It saves models/cwe_classes.json and adds `cwe_y` to every cached graph.
"""

from __future__ import annotations

import os
import json
from collections import Counter

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graphs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TOP_K = 5
OTHER = "Other"
IGNORE = -1          # label for "do not train the CWE head on this function"

# short human-readable names, used in the extension tooltip
CWE_NAMES = {
    "CWE-119": "buffer errors (memory bounds)",
    "CWE-20": "improper input validation",
    "CWE-399": "resource management errors",
    "CWE-264": "permissions / access control",
    "CWE-200": "information exposure",
    "CWE-125": "out-of-bounds read",
    "CWE-787": "out-of-bounds write",
    "CWE-416": "use after free",
    "CWE-476": "NULL pointer dereference",
    "CWE-190": "integer overflow",
    "CWE-189": "numeric errors",
    "CWE-362": "race condition",
    "CWE-284": "improper access control",
    "CWE-310": "cryptographic issues",
    OTHER: "other weakness type",
}


def has_real_cwe(value: str) -> bool:
    v = (value or "").strip()
    return v.upper().startswith("CWE-")


def choose_classes(train_df: pd.DataFrame, k: int = TOP_K) -> list[str]:
    """Top-k CWEs among labelled VULNERABLE training functions, plus 'Other'."""
    vuln = train_df[train_df["label"] == 1]
    counts = Counter(c for c in vuln["cwe"] if has_real_cwe(c))
    return [c for c, _ in counts.most_common(k)] + [OTHER]


def cwe_index(cwe: str, label: int, classes: list[str]) -> int:
    """Class index for one function, or IGNORE if it must not train the head."""
    if int(label) != 1 or not has_real_cwe(cwe):
        return IGNORE
    return classes.index(cwe) if cwe in classes else classes.index(OTHER)


def save_classes(classes: list[str]):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "cwe_classes.json"), "w") as fh:
        json.dump(classes, fh)


def load_classes() -> list[str] | None:
    path = os.path.join(MODELS_DIR, "cwe_classes.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def describe(cwe: str) -> str:
    return CWE_NAMES.get(cwe, cwe)


def main():
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    classes = choose_classes(train_df)
    save_classes(classes)
    print("CWE classes (from TRAIN only):", classes)

    for split in ["train", "val", "test"]:
        df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
        graphs = torch.load(os.path.join(GRAPH_DIR, f"{split}.pt"), weights_only=False)
        assert len(graphs) == len(df), f"{split}: graphs and rows are misaligned"

        labels = [cwe_index(c, y, classes) for c, y in zip(df["cwe"], df["label"])]
        for g, lab in zip(graphs, labels):
            g.cwe_y = torch.tensor([lab], dtype=torch.long)
        torch.save(graphs, os.path.join(GRAPH_DIR, f"{split}.pt"))

        used = [l for l in labels if l != IGNORE]
        counts = Counter(used)
        dist = ", ".join(f"{classes[i]}:{counts.get(i, 0)}" for i in range(len(classes)))
        print(f"  {split:<5}: {len(used):>4} labelled vulnerable functions | {dist}")

    # honest baseline: always predict the most common TRAIN class
    train_labels = [cwe_index(c, y, classes) for c, y in zip(train_df["cwe"], train_df["label"])]
    majority = Counter(l for l in train_labels if l != IGNORE).most_common(1)[0][0]
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    test_labels = [l for l in (cwe_index(c, y, classes)
                               for c, y in zip(test_df["cwe"], test_df["label"])) if l != IGNORE]
    acc = sum(l == majority for l in test_labels) / len(test_labels)
    print(f"\nMajority-class baseline (always '{classes[majority]}'): "
          f"test accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
