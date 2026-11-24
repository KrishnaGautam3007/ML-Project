"""
Analysis for the report: EDA, data-quality checks, size baselines, length
bands, bootstrap confidence intervals, calibration and learning curves.

Everything here uses the EXISTING data, the saved model and the saved training
log - no model is retrained. Results go to:
    results/analysis.json          every number quoted in the report
    docs/assets/an_*.png           figures used in the report

Run (after Phases 1-5 and src/calibrate_saved.py):
    python src/analysis.py
"""

from __future__ import annotations

import os
import re
import sys
import json
import hashlib

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             precision_recall_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "baselines"))

import calibration  # noqa: E402
from calibrate_saved import logits_for  # noqa: E402
from evaluate import precision_at_top_k_percent  # noqa: E402
from explain import load_model  # noqa: E402
from features import METRIC_NAMES, load_metric_stats  # noqa: E402
from gnn_data import load_graphs  # noqa: E402

PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "docs", "assets")
RNG = np.random.default_rng(0)

# ---- chart style (data-viz reference palette, light mode) ----
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SAFE_C, VULN_C = BLUE, ORANGE
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

OUT: dict = {}


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def h(title):
    print("\n" + "=" * 10 + " " + title + " " + "=" * 10)


# =============================================================================
# 1. data
# =============================================================================

splits = {s: pd.read_parquet(os.path.join(PROCESSED, f"{s}.parquet")) for s in ("train", "val", "test")}
sub = pd.concat([d.assign(split=s) for s, d in splits.items()], ignore_index=True)

# raw metric values: the cached graphs store z-scored metrics; undo the scaling
stats = load_metric_stats()
mean, std = np.array(stats["mean"]), np.array(stats["std"])
graphs = {s: load_graphs(s) for s in ("train", "val", "test")}
metrics_z = {s: np.vstack([g.metrics.numpy() for g in graphs[s]]) for s in graphs}
# all six metrics are counts, so round away floating-point leftovers (~1e-10)
metrics_raw = {s: np.round(metrics_z[s] * std + mean) for s in graphs}
sub_metrics = pd.DataFrame(np.vstack([metrics_raw[s] for s in ("train", "val", "test")]),
                           columns=METRIC_NAMES)
sub_metrics["label"] = sub["label"].to_numpy()

# =============================================================================
# 2. EDA: raw dataset quality
# =============================================================================
h("EDA: raw Big-Vul")
from datasets import load_dataset  # noqa: E402

ds = load_dataset("bstee615/bigvul")
raw = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
code = raw["func_before"].astype(str)
nlines = code.str.count("\n") + 1
norm_hash = code.map(lambda c: hashlib.sha1(re.sub(r"\s+", " ", c).strip().encode()).hexdigest())
grp = raw.assign(h=norm_hash).groupby("h")["vul"].agg(["count", "nunique"])
cwe_ok = raw["CWE ID"].astype(str).str.startswith("CWE-")
OUT["raw"] = {
    "rows": len(raw), "vulnerable": int(raw["vul"].sum()),
    "vulnerable_pct": round(100 * raw["vul"].mean(), 2),
    "projects": int(raw["project"].nunique()),
    "null_counts": {c: int(raw[c].isna().sum()) for c in raw.columns},
    "empty_code": int((code.str.strip() == "").sum()),
    "lines_median": float(nlines.median()), "lines_p95": float(nlines.quantile(0.95)),
    "lines_lt3": int((nlines < 3).sum()), "lines_gt500": int((nlines > 500).sum()),
    "unique_bodies": int(len(grp)), "duplicate_rows": int((grp["count"] - 1).sum()),
    "conflicting_label_groups": int((grp["nunique"] > 1).sum()),
    "conflicting_label_rows": int(grp.loc[grp["nunique"] > 1, "count"].sum()),
    "vulnerable_without_cwe": int(((raw["vul"] == 1) & ~cwe_ok).sum()),
    "safe_with_cwe": int(((raw["vul"] == 0) & cwe_ok).sum()),
    "after_length_filter": 213407, "after_dedup": 160888,   # printed by src/dataset.py
}
print(json.dumps(OUT["raw"], indent=1))

# =============================================================================
# 3. EDA: the 30k subset the model uses
# =============================================================================
h("EDA: 30k subset")
per_split = {}
for s, d in splits.items():
    vc = d.groupby("project").size().sort_values(ascending=False)
    per_split[s] = {
        "functions": len(d), "vulnerable": int(d.label.sum()),
        "vulnerable_pct": round(100 * d.label.mean(), 2), "projects": int(d.project.nunique()),
        "median_lines_safe": float(d.n_lines[d.label == 0].median()),
        "median_lines_vuln": float(d.n_lines[d.label == 1].median()),
        "largest_project": vc.index[0], "largest_project_functions": int(vc.iloc[0]),
        "largest_project_share_pct": round(100 * vc.iloc[0] / len(d), 1),
        "top5_share_pct": round(100 * vc.iloc[:5].sum() / len(d), 1),
        "median_project_size": float(vc.median()),
    }
proj = sub.groupby("project")["label"].agg(["size", "sum"])
OUT["subset"] = {
    "functions": len(sub), "vulnerable": int(sub.label.sum()),
    "vulnerable_pct": round(100 * sub.label.mean(), 2),
    "projects": int(sub.project.nunique()),
    "projects_without_vulnerable": int((proj["sum"] == 0).sum()),
    "per_split": per_split,
    "pos_weight": round((len(splits["train"]) - splits["train"].label.sum())
                        / splits["train"].label.sum(), 2),
}
vuln = sub[sub.label == 1]
cw = vuln["cwe"].where(vuln["cwe"].str.startswith("CWE-"), "no CWE label")
OUT["subset"]["cwe_top"] = cw.value_counts().head(11).to_dict()
OUT["metric_medians"] = {
    m: {"safe": float(sub_metrics.loc[sub_metrics.label == 0, m].median()),
        "vulnerable": float(sub_metrics.loc[sub_metrics.label == 1, m].median())}
    for m in METRIC_NAMES}
corr = sub_metrics.corr(method="spearman")
OUT["metric_spearman"] = corr.round(3).to_dict()
print(json.dumps(OUT["subset"], indent=1, default=str))

# ---- EDA figures ----
# (a) class balance per split
fig, ax = plt.subplots(figsize=(6.4, 2.3))
names = ["Raw Big-Vul", "Train", "Validation", "Test"]
vals = [OUT["raw"]["vulnerable_pct"]] + [per_split[s]["vulnerable_pct"] for s in ("train", "val", "test")]
counts = [f"{OUT['raw']['vulnerable']:,} / {OUT['raw']['rows']:,}"] + [
    f"{per_split[s]['vulnerable']:,} / {per_split[s]['functions']:,}" for s in ("train", "val", "test")]
y = np.arange(len(names))[::-1]
ax.barh(y, vals, height=0.55, color=VULN_C)
for yi, v, c in zip(y, vals, counts):
    ax.text(v + 0.1, yi, f"{v:.1f}%   ({c})", va="center", color=INK, fontsize=8.5)
ax.set_yticks(y, names)
ax.set_xlim(0, 10)
ax.set_xlabel("vulnerable functions (%)")
ax.set_title("Class balance: about 1 in 19 functions is vulnerable", loc="left")
style(ax, "x")
save(fig, "an_class_balance.png")

# (b) function length by label (log scale)
fig, ax = plt.subplots(figsize=(6.4, 2.6))
# whole-number bin edges: line counts are integers, so log-spaced bins narrower
# than one line would leave empty gaps at the short end
bins = np.unique(np.round(np.logspace(np.log10(3), np.log10(501), 32))) - 0.5
for lab, c, nm in ((0, SAFE_C, "safe"), (1, VULN_C, "vulnerable")):
    d = sub.n_lines[sub.label == lab]
    ax.hist(d, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=c)
    ax.hist(d, bins=bins, density=True, histtype="step", linewidth=1.6, color=c,
            label=f"{nm} (median {d.median():.0f} lines)")
ax.set_xscale("log")
ax.set_xlabel("lines of code (log scale)")
ax.set_ylabel("density")
ax.set_title("Vulnerable functions are much longer than safe ones", loc="left")
ax.legend(loc="upper right")
style(ax)
save(fig, "an_length.png")

# (c) CWE types among vulnerable functions
fig, ax = plt.subplots(figsize=(6.4, 2.9))
top = OUT["subset"]["cwe_top"]
labels, nums = list(top.keys())[::-1], list(top.values())[::-1]
colors = [MUTED if lab == "no CWE label" else VULN_C for lab in labels]
ax.barh(labels, nums, color=colors, height=0.6)
for i, n in enumerate(nums):
    ax.text(n + 4, i, str(n), va="center", fontsize=8.5, color=INK)
ax.set_xlabel("vulnerable functions in the 30k subset")
ax.set_title("Most common vulnerability types (CWE)", loc="left")
style(ax, "x")
save(fig, "an_cwe.png")

# (d) projects: who dominates each split
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
for ax, s in zip(axes, ("train", "val", "test")):
    d = splits[s]
    vc = d.groupby("project").size().sort_values(ascending=False)
    top3 = vc.iloc[:3]
    shares = list(100 * top3 / len(d)) + [100 * vc.iloc[3:].sum() / len(d)]
    labs = list(top3.index) + [f"{len(vc) - 3} other projects"]
    yy = np.arange(4)[::-1]
    ax.barh(yy, shares, color=[BLUE, BLUE, BLUE, MUTED], height=0.6)
    for yi, v, lab in zip(yy, shares, labs):
        ax.text(1, yi + 0.36, lab, fontsize=7.5, color=INK2)
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_title(f"{s.capitalize()} ({per_split[s]['projects']} projects)", fontsize=9, loc="left")
    style(ax, "x")
axes[0].set_ylabel("share of functions")
fig.suptitle("Train is dominated by two large codebases; val/test are many small projects",
             x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK, y=1.05)
save(fig, "an_projects.png")

# (e) metric distributions by label
fig, axes = plt.subplots(1, 6, figsize=(6.8, 2.3))
nice = {"loc": "lines of code", "cyclomatic": "cyclomatic", "pointer_ops": "pointer ops",
        "risky_calls": "risky calls", "max_nesting": "max nesting", "n_params": "parameters"}
for ax, m in zip(axes, METRIC_NAMES):
    if m == "risky_calls":
        # mostly zero, so a box plot collapses; show how often there is any risky call
        share = [100 * (sub_metrics.loc[sub_metrics.label == lab, m] > 0).mean() for lab in (0, 1)]
        ax.bar([1, 2], share, color=(SAFE_C, VULN_C), alpha=0.75, width=0.55)
        for x, v in zip((1, 2), share):
            ax.text(x, v + 0.5, f"{v:.0f}%", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks([1, 2], ["safe", "vuln"], fontsize=7.5)
        ax.set_title("has a risky call", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("% of functions", fontsize=7.5)
        style(ax)
        continue
    data = [np.log1p(sub_metrics.loc[sub_metrics.label == lab, m]) for lab in (0, 1)]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.4})
    for patch, c in zip(bp["boxes"], (SAFE_C, VULN_C)):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(c)
    ax.set_xticks([1, 2], ["safe", "vuln"], fontsize=7.5)
    ax.set_title(nice[m], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    style(ax)
axes[0].set_ylabel("log(1 + value)")
fig.suptitle("Code metrics by label (boxes = middle 50%, line = median)",
             x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK, y=1.07)
fig.tight_layout()
save(fig, "an_metrics.png")

# (f) Spearman correlation between metrics and the label
fig, ax = plt.subplots(figsize=(4.4, 3.6))
cols = METRIC_NAMES + ["label"]
cm = corr.loc[cols, cols].to_numpy()
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
tick = [nice.get(c, "vulnerable (label)") for c in cols]
ax.set_xticks(range(len(cols)), tick, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(cols)), tick, fontsize=8)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if cm[i, j] > 0.6 else INK)
ax.set_title("Spearman correlation between metrics", loc="left")
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save(fig, "an_corr.png")

# =============================================================================
# 4. models and baselines on the SAME test set
# =============================================================================
h("Scores on test")
test = splits["test"]
y_test = test.label.to_numpy()

model, vocab, _, calib, thr = load_model()
ckpt_prev = torch.load(os.path.join(ROOT, "models", "hybrid_gnn.pt"), map_location="cpu",
                       weights_only=False).get("previous_calibration")
z_test, y_check = logits_for(model, "test", vocab)
assert (y_check == y_test).all(), "graph/parquet order mismatch"
p_model = calibration.apply(calib, z_test)
p_model_old = calibration.apply(ckpt_prev, z_test) if ckpt_prev else None

from run_baselines import run_flawfinder, run_tfidf_lr, run_risky_api  # noqa: E402

scores = {"Hybrid GraphSAGE (ours)": p_model}
_, scores["Flawfinder (level >= 2)"] = run_flawfinder(test["code"].tolist())
_, scores["TF-IDF + Logistic Regression"] = run_tfidf_lr(splits["train"], test)
_, scores["Risky-API rule"] = run_risky_api(test["code"].tolist())
scores["Lines of code only"] = test.n_lines.to_numpy().astype(float)
ytr = splits["train"].label.to_numpy()
lr6 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(metrics_z["train"], ytr)
scores["6 metrics only (LogReg)"] = lr6.predict_proba(metrics_z["test"])[:, 1]
OUT["metrics_lr_coef"] = dict(zip(METRIC_NAMES, lr6.coef_[0].round(3).tolist()))
# the same length baseline on VALIDATION (shows the val/test gap is a property of the splits)
OUT["loc_val_ap"] = round(float(average_precision_score(splits["val"].label,
                                                         splits["val"].n_lines.astype(float))), 3)


def bootstrap(score_a, score_b=None, n=2000):
    """95% CI of AP and P@10 (or of their difference a-b, paired)."""
    aps, p10s = [], []
    idx_all = np.arange(len(y_test))
    for _ in range(n):
        idx = RNG.choice(idx_all, size=len(idx_all), replace=True)
        yy = y_test[idx]
        if yy.sum() == 0:
            continue
        a = average_precision_score(yy, score_a[idx])
        b = precision_at_top_k_percent(yy, score_a[idx])
        if score_b is not None:
            a -= average_precision_score(yy, score_b[idx])
            b -= precision_at_top_k_percent(yy, score_b[idx])
        aps.append(a)
        p10s.append(b)
    q = lambda v: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]  # noqa: E731
    return q(aps), q(p10s)


OUT["test_scores"] = {}
for name, s in scores.items():
    ci_ap, ci_p10 = bootstrap(np.asarray(s, dtype=float))
    OUT["test_scores"][name] = {
        "ap": round(float(average_precision_score(y_test, s)), 3), "ap_ci95": ci_ap,
        "p_at_10": round(float(precision_at_top_k_percent(y_test, s)), 3), "p_at_10_ci95": ci_p10}
    print(name, OUT["test_scores"][name])

OUT["paired_differences"] = {}
for other in ("Lines of code only", "6 metrics only (LogReg)", "TF-IDF + Logistic Regression",
              "Risky-API rule"):
    d_ap, d_p10 = bootstrap(p_model, np.asarray(scores[other], dtype=float))
    OUT["paired_differences"][f"model minus {other}"] = {"ap_ci95": d_ap, "p_at_10_ci95": d_p10}
print(json.dumps(OUT["paired_differences"], indent=1))

# confusion matrix at the deployed threshold
tn, fp, fn, tp = confusion_matrix(y_test, (p_model >= thr).astype(int)).ravel()
OUT["confusion_matrix"] = {"threshold": round(thr, 4), "tn": int(tn), "fp": int(fp),
                           "fn": int(fn), "tp": int(tp)}
print("confusion:", OUT["confusion_matrix"])

# PR curves
fig, ax = plt.subplots(figsize=(6.4, 3.4))
curves = [("Hybrid GraphSAGE (ours)", BLUE, 2.2), ("Lines of code only", ORANGE, 1.6),
          ("6 metrics only (LogReg)", AQUA, 1.6), ("TF-IDF + Logistic Regression", YELLOW, 1.6),
          ("Risky-API rule", MAGENTA, 1.6)]
for name, c, lw in curves:
    pr, rc, _ = precision_recall_curve(y_test, scores[name])
    ax.plot(rc, pr, color=c, linewidth=lw, label=f"{name} (AP {OUT['test_scores'][name]['ap']:.3f})",
            drawstyle="steps-post")
ax.axhline(y_test.mean(), color=INK2, linestyle="--", linewidth=1)
ax.text(0.30, y_test.mean() - 0.03, f"random = {y_test.mean():.3f}", ha="left", fontsize=8, color=INK2)
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.6)
ax.set_title("Precision-recall curves on the test set", loc="left")
ax.legend(fontsize=7.8, loc="upper right")
style(ax, "both")
save(fig, "an_pr_curves.png")

# =============================================================================
# 5. performance inside length bands (does the model add anything beyond size?)
# =============================================================================
h("Length bands")
q = np.quantile(test.n_lines, [0, 0.25, 0.5, 0.75, 1.0])
bands = []
for i in range(4):
    lo, hi = q[i], q[i + 1]
    m = (test.n_lines >= lo) & ((test.n_lines < hi) if i < 3 else (test.n_lines <= hi))
    m = m.to_numpy()
    row = {"lines": f"{int(lo)}-{int(hi)}", "functions": int(m.sum()), "vulnerable": int(y_test[m].sum()),
           "rate": round(float(y_test[m].mean()), 3)}
    for name in ("Hybrid GraphSAGE (ours)", "Lines of code only", "6 metrics only (LogReg)"):
        row[name] = round(float(average_precision_score(y_test[m], np.asarray(scores[name])[m])), 3) \
            if y_test[m].sum() > 0 else None
    bands.append(row)
    print(row)
OUT["length_bands"] = bands
OUT["spearman_model_vs_loc"] = round(float(pd.Series(p_model).corr(test.n_lines.reset_index(drop=True),
                                                                   method="spearman")), 3)
print("spearman(model score, LOC):", OUT["spearman_model_vs_loc"])

# =============================================================================
# 6. calibration: before (temperature) vs after (Platt)
# =============================================================================
h("Calibration")


def reliability(p, n_bins=10):
    order = np.argsort(p)
    return [(float(p[c].mean()), float(y_test[c].mean())) for c in np.array_split(order, n_bins)]


cal = {}
for nm, p in (("temperature (before)", p_model_old), ("Platt (after)", p_model)):
    if p is None:
        continue
    cal[nm] = {
        "min": round(float(p.min()), 3), "max": round(float(p.max()), 3),
        "mean": round(float(p.mean()), 3), "true_rate": round(float(y_test.mean()), 3),
        "ece": round(calibration.expected_calibration_error(p, y_test), 3),
        "bands": {"Low": int((p < 0.4).sum()), "Medium": int(((p >= 0.4) & (p <= 0.7)).sum()),
                  "High": int((p > 0.7).sum())},
        "reliability": reliability(p)}
from explain import band  # noqa: E402

OUT["calibration"] = {**cal, "platt": calib, "previous": ckpt_prev,
                      "threshold_bands": dict(pd.Series([band(float(p), thr) for p in p_model])
                                              .value_counts())}
print("threshold-anchored bands on test:", OUT["calibration"]["threshold_bands"])
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "reliability"} for k, v in cal.items()}, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
ax = axes[0]
for nm, c in (("temperature (before)", MUTED), ("Platt (after)", BLUE)):
    r = np.array(cal[nm]["reliability"])
    ax.plot(r[:, 0], r[:, 1], marker="o", markersize=4, color=c, linewidth=1.8,
            label=f"{nm}, ECE {cal[nm]['ece']:.3f}")
ax.plot([0, 0.7], [0, 0.7], color=INK2, linestyle="--", linewidth=1)
ax.text(0.45, 0.5, "perfect", rotation=38, fontsize=7.5, color=INK2)
ax.set_xlim(0, 0.7)
ax.set_ylim(0, 0.7)
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed vulnerable rate")
ax.set_title("Reliability (10 equal-size bins)", loc="left")
ax.legend(fontsize=7.5, loc="upper left")
style(ax, "both")
ax = axes[1]
b = np.linspace(0, 0.7, 36)
ax.hist(p_model_old, bins=b, color=MUTED, alpha=0.6, label="temperature (before)")
ax.hist(p_model, bins=b, color=BLUE, alpha=0.6, label="Platt (after)")
ax.set_xlabel("predicted probability")
ax.set_ylabel("test functions")
ax.set_title("Where the probabilities fall", loc="left")
ax.legend(fontsize=7.5)
style(ax)
fig.tight_layout()
save(fig, "an_calibration.png")

# =============================================================================
# 7. learning curves of the reported run (from the saved training log)
# =============================================================================
h("Learning curves")
log = open(os.path.join(RESULTS, "original_training_log.txt"), encoding="utf-8", errors="ignore").read()
curves_lc, current = {}, None
for line in log.splitlines():
    m = re.search(r"--- Training: (.+?) ---", line)
    if m:
        current = m.group(1)
        curves_lc[current] = []
    m = re.search(r"epoch\s+(\d+) \| loss ([\d.]+) \| val PR-AUC ([\d.]+)", line)
    if m and current:
        curves_lc[current].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
OUT["learning_curves"] = curves_lc
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
palette = {"Hybrid GraphSAGE (main)": BLUE, "Ablation: AST edges only": ORANGE,
           "Ablation: no metrics": AQUA, "Ablation: GCN (not SAGE)": YELLOW}
for name, pts in curves_lc.items():
    pts = np.array(pts)
    c = palette.get(name, MUTED)
    axes[0].plot(pts[:, 0], pts[:, 1], color=c, marker="o", markersize=3.5, linewidth=1.6, label=name)
    axes[1].plot(pts[:, 0], pts[:, 2], color=c, marker="o", markersize=3.5, linewidth=1.6, label=name)
    best = pts[np.argmax(pts[:, 2])]
    axes[1].plot(best[0], best[2], marker="o", markersize=8, markerfacecolor="none", color=c)
axes[0].set_title("Training loss", loc="left")
axes[1].set_title("Validation PR-AUC (circle = kept epoch)", loc="left")
for ax in axes:
    ax.set_xlabel("epoch")
    style(ax)
handles, labels_lc = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_lc, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "an_learning_curves.png")

# =============================================================================
# 8. input/output demo for the report (backend/sample.c, function copy_input)
# =============================================================================
from explain import explain_function  # noqa: E402
from parsing import iter_functions  # noqa: E402

sample = open(os.path.join(ROOT, "backend", "sample.c"), encoding="utf-8").read()
for fn in iter_functions(sample):
    if fn.name == "copy_input":
        exp = explain_function(fn.code, model, vocab, stats, calib, thr)
        tl = exp["top_lines"][0]
        OUT["demo"] = {"name": fn.name, "start_line": fn.start_line, "end_line": fn.end_line,
                       "risk_score": exp["risk_score"], "band": exp["band"],
                       "top_line": fn.start_line + tl["line"] - 1, "reason": tl["reason"],
                       "text": tl["text"]}
print("demo:", OUT.get("demo"))

os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "analysis.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print("\nSaved results/analysis.json and docs/assets/an_*.png")
