"""
Phase 2 - Three baselines the GNN must beat.

  1. Flawfinder      - a classic rule/pattern scanner. A function is "risky" if
                       Flawfinder reports any hit of severity level >= 2 in it.
  2. TF-IDF + LogReg - treat the code as plain text, turn it into word counts,
                       and train a simple Logistic Regression. This is the
                       "code as text" baseline the whole project argues against.
  3. Risky-API rule  - flag any function that calls a known-dangerous C function
                       (strcpy, strcat, gets, sprintf, memcpy, scanf) or uses
                       malloc / free.

Run:
    python baselines/run_baselines.py

It prints one table comparing all three on the TEST set.
"""

from __future__ import annotations

import os
import re
import sys
import csv
import shutil
import tempfile
import subprocess

import numpy as np
import pandas as pd

# let us import src/evaluate.py no matter where we run from
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from evaluate import evaluate, print_table  # noqa: E402

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ---------------------------------------------------------------------------
# Baseline 1: Flawfinder
# ---------------------------------------------------------------------------

def run_flawfinder(codes: list[str], min_level: int = 2):
    """Run Flawfinder over every function and return (y_pred, y_score).

    We write each function to its own tiny .c file in a temp folder, then run
    Flawfinder ONCE over the whole folder (one process, much faster than one
    process per function). A function is flagged if it has any hit of
    severity >= min_level; its score is the maximum hit level found (0 if none),
    which lets us rank functions for PR-AUC / P@10%.
    """
    n = len(codes)
    tmp = tempfile.mkdtemp(prefix="ff_")
    try:
        for i, code in enumerate(codes):
            with open(os.path.join(tmp, f"{i}.c"), "w", encoding="utf-8", errors="ignore") as fh:
                fh.write(code)

        # python -m flawfinder works regardless of PATH on Windows
        cmd = [sys.executable, "-m", "flawfinder", "--csv", f"--minlevel={min_level}", tmp]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        scores = np.zeros(n, dtype=float)
        reader = csv.DictReader(proc.stdout.splitlines())
        for row in reader:
            fname = row.get("File", "")
            stem = os.path.splitext(os.path.basename(fname))[0]
            if not stem.isdigit():
                continue
            idx = int(stem)
            try:
                level = float(row.get("Level", 0))
            except ValueError:
                level = 0.0
            if 0 <= idx < n:
                scores[idx] = max(scores[idx], level)

        y_pred = (scores >= min_level).astype(int)
        return y_pred, scores
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Baseline 2: TF-IDF + Logistic Regression ("code as plain text")
# ---------------------------------------------------------------------------

def run_tfidf_lr(train_df: pd.DataFrame, test_df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # \b\w+\b keeps identifiers and numbers (incl. single chars like 'i', 'n').
    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
    )
    X_train = vec.fit_transform(train_df["code"])
    X_test = vec.transform(test_df["code"])

    # class_weight="balanced" tells LR to pay more attention to the rare
    # vulnerable class instead of ignoring it.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_df["label"])

    y_score = clf.predict_proba(X_test)[:, 1]   # probability of "vulnerable"
    y_pred = clf.predict(X_test)
    return y_pred, y_score


# ---------------------------------------------------------------------------
# Baseline 3: Risky-API rule
# ---------------------------------------------------------------------------

RISKY_APIS = ["strcpy", "strcat", "gets", "sprintf", "memcpy", "scanf", "malloc", "free"]
# match e.g. `strcpy(` but not `mystrcpy(` thanks to the word boundary
RISKY_RE = re.compile(r"\b(" + "|".join(RISKY_APIS) + r")\s*\(")


def run_risky_api(codes: list[str]):
    """Flag any function that calls a risky API. Score = number of such calls."""
    scores = np.array([len(RISKY_RE.findall(c)) for c in codes], dtype=float)
    y_pred = (scores >= 1).astype(int)
    return y_pred, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

    y_true = test_df["label"].to_numpy()
    test_codes = test_df["code"].tolist()

    print(f"Test set: {len(test_df):,} functions, "
          f"{int(y_true.sum()):,} vulnerable ({100*y_true.mean():.1f}%)\n")

    results = []

    print("Running Flawfinder (this is the slowest one)...")
    ff_pred, ff_score = run_flawfinder(test_codes)
    results.append(evaluate("Flawfinder (level>=2)", y_true, ff_pred, ff_score))

    print("Training TF-IDF + Logistic Regression...")
    lr_pred, lr_score = run_tfidf_lr(train_df, test_df)
    results.append(evaluate("TF-IDF + LogReg", y_true, lr_pred, lr_score))

    print("Applying risky-API rule...\n")
    ra_pred, ra_score = run_risky_api(test_codes)
    results.append(evaluate("Risky-API rule", y_true, ra_pred, ra_score))

    print("=== Phase 2 baselines (TEST set) ===")
    print_table(results)
    print("\nReminder: a random model scores PR-AUC ~= the vulnerable rate "
          f"(~{y_true.mean():.3f}).")


if __name__ == "__main__":
    main()
