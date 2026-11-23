"""
Phase 1 - Dataset download and cleaning.

What this file does, in plain language:
  1. Downloads the Big-Vul dataset (C/C++ functions labelled vulnerable / safe,
     each row also carries the project name, the CWE id, and the lines the
     security fix touched).
  2. Cleans it:
       - removes exact-duplicate functions (same code after we strip whitespace),
       - drops functions shorter than 3 lines or longer than 500 lines,
       - keeps the project name on every row.
  3. Splits the data BY PROJECT into train / val / test (~80 / 10 / 10) so that
     the same project never appears in two splits (see the note in split_by_project
     for *why* this matters).
  4. Prints statistics and saves the three cleaned splits to data/processed/.

Run it with:
    python src/dataset.py --subset 30000        # cap the size on a laptop
    python src/dataset.py                        # use everything

This script only needs the `datasets`, `pandas` and `pyarrow` libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter

import pandas as pd

# ----------------------------------------------------------------------------
# Where things live
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# The Hugging Face dataset id for Big-Vul. This is a widely-mirrored copy of the
# original Big-Vul CSV released by Fan et al. (2020).
HF_DATASET_ID = "bstee615/bigvul"

# Minimum / maximum function length (in lines) we keep.
MIN_LINES = 3
MAX_LINES = 500


# ----------------------------------------------------------------------------
# Step 0 - helpers to find the right columns
# ----------------------------------------------------------------------------
# Different mirrors of Big-Vul name their columns slightly differently, so we
# look for the first name that exists out of a list of likely candidates.

def pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True):
    """Return the first column name in `candidates` that exists in df."""
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(
            f"None of these columns were found: {candidates}\n"
            f"Columns that DO exist: {list(df.columns)}"
        )
    return None


# ----------------------------------------------------------------------------
# Step 1 - download
# ----------------------------------------------------------------------------

def download_bigvul() -> pd.DataFrame:
    """Download Big-Vul from Hugging Face and return it as one pandas DataFrame."""
    from datasets import load_dataset, concatenate_datasets

    print(f"Downloading '{HF_DATASET_ID}' from Hugging Face (first run only)...")
    ds = load_dataset(HF_DATASET_ID)

    # The dataset may already come split (train/validation/test). We merge every
    # split back into one table and do our OWN project-based split afterwards.
    parts = [ds[s] for s in ds.keys()]
    merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    df = merged.to_pandas()
    print(f"  downloaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


# ----------------------------------------------------------------------------
# Step 2 - normalise into a tidy table
# ----------------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns we care about to a fixed, simple set of names."""
    code_col = pick_column(df, ["func_before", "func", "function", "code"])
    label_col = pick_column(df, ["vul", "target", "label", "is_vulnerable"])
    project_col = pick_column(df, ["project", "repo", "repository"], required=False)
    cwe_col = pick_column(df, ["CWE ID", "cwe", "cwe_id", "CWE"], required=False)
    # The FIXED version of the function. This mirror has no explicit flaw-line
    # column, so in Phase 6 we recover the vulnerable lines by diffing
    # func_before (our `code`) against func_after.
    after_col = pick_column(df, ["func_after", "func_fixed", "fixed_func"], required=False)

    out = pd.DataFrame()
    out["code"] = df[code_col].astype(str)
    # Force the label to a clean 0/1 integer.
    out["label"] = df[label_col].astype(int).clip(0, 1)
    out["project"] = df[project_col].astype(str) if project_col else "unknown"
    out["cwe"] = df[cwe_col].astype(str) if cwe_col else ""
    out["func_after"] = df[after_col].astype(str) if after_col else ""
    return out


# ----------------------------------------------------------------------------
# Step 3 - cleaning
# ----------------------------------------------------------------------------

def normalise_whitespace(code: str) -> str:
    """Collapse all runs of whitespace to a single space, for duplicate detection.

    Two functions that differ only in indentation / blank lines should count as
    the same function.
    """
    return re.sub(r"\s+", " ", code).strip()


def code_hash(code: str) -> str:
    """A short fingerprint of a function, ignoring whitespace differences."""
    return hashlib.sha1(normalise_whitespace(code).encode("utf-8")).hexdigest()


def line_count(code: str) -> int:
    return code.count("\n") + 1


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # a) drop empty / missing code
    df = df[df["code"].str.strip().str.len() > 0].copy()

    # b) length filter
    df["n_lines"] = df["code"].map(line_count)
    df = df[(df["n_lines"] >= MIN_LINES) & (df["n_lines"] <= MAX_LINES)].copy()
    after_len = len(df)

    # c) remove exact duplicates (whitespace-insensitive)
    df["_hash"] = df["code"].map(code_hash)
    df = df.drop_duplicates(subset="_hash", keep="first").copy()
    df = df.drop(columns="_hash")
    after_dup = len(df)

    print("Cleaning:")
    print(f"  start                         : {before:,}")
    print(f"  after length filter (3..500)  : {after_len:,}")
    print(f"  after de-duplication          : {after_dup:,}")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Step 4 - split BY PROJECT
# ----------------------------------------------------------------------------

def split_by_project(df: pd.DataFrame, seed: int = 42):
    """Split into train/val/test so no project is in two splits.

    WHY NOT a plain random split?
      Functions from the same project share coding style, helper functions and
      naming. If we split rows randomly, near-identical functions from one
      project land in both train AND test. The model then looks great in testing
      but only because it memorised that project's style - it "leaked". Splitting
      by project forces the model to generalise to code it has never seen, which
      is the realistic setting when we scan a brand-new GitHub repo.

    HOW we balance the sizes:
      Big-Vul has a few huge projects and many tiny ones, so simply giving 80%
      of the *projects* to train can hand train 95% of the *functions* and leave
      the test set tiny. Instead we use a simple greedy packer: take projects
      biggest-first, and drop each one into whichever split is currently furthest
      below its target share (80/10/10 of functions). Projects stay whole, so
      there is still zero leakage, but the function counts come out close to
      80/10/10.
    """
    # size (number of functions) of each project
    sizes = df.groupby("project").size().to_dict()
    # biggest projects first; tie-break by name so the result is reproducible
    projects = sorted(sizes, key=lambda p: (-sizes[p], p))

    total = len(df)
    targets = {"train": 0.8 * total, "val": 0.1 * total, "test": 0.1 * total}
    current = {"train": 0, "val": 0, "test": 0}
    members = {"train": set(), "val": set(), "test": set()}

    for p in projects:
        # split that is furthest below its target gets this project
        split = max(targets, key=lambda s: targets[s] - current[s])
        members[split].add(p)
        current[split] += sizes[p]

    train = df[df["project"].isin(members["train"])].reset_index(drop=True)
    val = df[df["project"].isin(members["val"])].reset_index(drop=True)
    test = df[df["project"].isin(members["test"])].reset_index(drop=True)
    return train, val, test


# ----------------------------------------------------------------------------
# Step 5 - optional sub-sampling (keep the class ratio)
# ----------------------------------------------------------------------------

def subsample(df: pd.DataFrame, target: int, seed: int = 42) -> pd.DataFrame:
    """Shrink to ~`target` rows while keeping the vulnerable / safe ratio."""
    if target <= 0 or len(df) <= target:
        return df
    frac = target / len(df)
    # group by label so both classes shrink by the same fraction
    out = (
        df.groupby("label", group_keys=False)[df.columns.tolist()]
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )
    return out


# ----------------------------------------------------------------------------
# Step 6 - statistics
# ----------------------------------------------------------------------------

def print_stats(name: str, df: pd.DataFrame):
    n = len(df)
    vuln = int(df["label"].sum())
    pct = (100.0 * vuln / n) if n else 0.0
    print(f"  {name:<6}: {n:>7,} functions | vulnerable {vuln:>6,} ({pct:5.1f}%) "
          f"| projects {df['project'].nunique():>4}")


def print_top_cwes(df: pd.DataFrame, k: int = 10):
    cwes = [c for c in df["cwe"].tolist() if c and c.lower() != "nan"]
    counter = Counter(cwes)
    print(f"\nTop {k} CWEs (whole dataset):")
    for cwe, count in counter.most_common(k):
        print(f"  {cwe:<16} {count:>6,}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: build cleaned Big-Vul splits")
    parser.add_argument("--subset", type=int, default=0,
                        help="cap total functions (e.g. 30000). 0 = use everything")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. download (cached after the first run)
    raw = download_bigvul()

    # 2. tidy columns
    df = normalise(raw)

    # 3. clean
    df = clean(df)

    # 4. optional subset (before splitting, keeps class ratio)
    if args.subset:
        df = subsample(df, args.subset, seed=args.seed)
        print(f"\nSub-sampled to {len(df):,} functions (kept class ratio).")

    # 5. show global info
    print_top_cwes(df)

    # 6. split by project
    train, val, test = split_by_project(df, seed=args.seed)

    print("\nSplit sizes (by project, no overlap):")
    print_stats("train", train)
    print_stats("val", val)
    print_stats("test", test)

    # 7. save
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        part.to_parquet(path, index=False)
        print(f"saved {name} -> {path}")

    print("\nPhase 1 done.")


if __name__ == "__main__":
    main()
