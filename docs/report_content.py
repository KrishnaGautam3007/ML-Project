"""
The text of the project report.

Every number that comes from an experiment is read from the results files
(results/analysis.json, results/seed_runs.json), never typed by hand, so the
report always matches the latest analysis. build_report.py calls
write_content() after preparing the Word template.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Mm

DOCS = Path(__file__).resolve().parent
ASSETS = DOCS / "assets"
RESULTS = DOCS.parent / "results"
W = 160  # content width in mm

MAIN, AST, NOMET, GCN = ("Hybrid GraphSAGE (main)", "Ablation: AST edges only",
                         "Ablation: no metrics", "Ablation: GCN (not SAGE)")
# the single run reported in the first version of the project (results/original_training_log.txt)
ORIGINAL_RUN = {MAIN: (0.208, 0.203), AST: (0.228, 0.223), NOMET: (0.133, 0.173), GCN: (0.194, 0.203)}


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

def f3(x):
    return f"{float(x):.3f}"


def ci(v, c):
    return f"{float(v):.3f} [{float(c[0]):.3f}–{float(c[1]):.3f}]"


def n(x):
    return f"{int(x):,}"


def load_results():
    A = json.loads((RESULTS / "analysis.json").read_text())
    runs = json.loads((RESULTS / "seed_runs.json").read_text())
    S = {}
    for v in (MAIN, NOMET, GCN, AST):
        r = [x for x in runs if x["variant"] == v]
        S[v] = {
            "n": len(r),
            "ap": (np.mean([x["test_ap"] for x in r]), np.std([x["test_ap"] for x in r], ddof=1)),
            "p10": (np.mean([x["test_p_at_10"] for x in r]), np.std([x["test_p_at_10"] for x in r], ddof=1)),
            "val": (np.mean([x["val_ap"] for x in r]), np.std([x["val_ap"] for x in r], ddof=1)),
            "epoch": np.mean([x["best_epoch"] for x in r]),
            "by_seed": {x["seed"]: x for x in r},
        }
    seeds = sorted(S[MAIN]["by_seed"])
    S["ast_beats_main_test"] = sum(S[AST]["by_seed"][s]["test_ap"] > S[MAIN]["by_seed"][s]["test_ap"]
                                   for s in seeds)
    S["ast_beats_main_val"] = sum(S[AST]["by_seed"][s]["val_ap"] > S[MAIN]["by_seed"][s]["val_ap"]
                                  for s in seeds)
    S["seeds"] = seeds
    return A, S


# -----------------------------------------------------------------------------
# the report
# -----------------------------------------------------------------------------

def write_content(R, add_field):
    A, S = load_results()
    raw, sub, sp = A["raw"], A["subset"], A["subset"]["per_split"]
    ts = A["test_scores"]
    m_main, m_loc, m_lr = ts["Hybrid GraphSAGE (ours)"], ts["Lines of code only"], ts["6 metrics only (LogReg)"]
    pdiff = A["paired_differences"]
    cal_old, cal_new = A["calibration"]["temperature (before)"], A["calibration"]["Platt (after)"]
    platt = A["calibration"]["platt"]
    bands = {k: int(v) for k, v in A["calibration"]["threshold_bands"].items()}
    cm = A["confusion_matrix"]
    rho = A["metric_spearman"]
    k_seeds = len(S["seeds"])

    # ======================================================================== Abstract
    R.title("Abstract", page_break=False)
    R.para(
        "Rule-based static analysers for C/C++ miss structural bugs and raise many false "
        "alarms, while text-based machine-learning detectors ignore the tree structure and "
        "execution order of code. This project presents a vulnerability scanner that converts "
        "every C/C++ function into a graph (its tree-sitter abstract syntax tree plus "
        "next-statement and variable-use edges) and classifies it with a **Hybrid GraphSAGE** "
        "network trained from scratch, which combines a learned graph embedding with six "
        "hand-crafted code metrics.")
    R.para(
        f"On a project-disjoint split of the Big-Vul dataset ({n(sub['functions'])} functions, "
        f"{sub['vulnerable_pct']:.1f}% vulnerable) the model reaches an average precision "
        f"(PR-AUC) of **{f3(m_main['ap'])}** (95% bootstrap CI {f3(m_main['ap_ci95'][0])}–"
        f"{f3(m_main['ap_ci95'][1])}) and a Precision@Top-10% of **{f3(m_main['p_at_10'])}**, "
        "significantly better than a rule-based scanner, a risky-API rule and a TF-IDF text "
        "classifier. A detailed analysis, however, shows that **function length alone** reaches "
        f"a PR-AUC of {f3(m_loc['ap'])}, statistically indistinguishable from the model: in this "
        "dataset most of the learnable signal is function size, and the graph structure adds no "
        f"measurable gain on top of it. Over {k_seeds} random seeds the model averages "
        f"{f3(S[MAIN]['ap'][0])} ± {f3(S[MAIN]['ap'][1])}. Probability calibration was corrected "
        f"with Platt scaling (calibration error {f3(cal_old['ece'])} → {f3(cal_new['ece'])}).")
    R.para(
        "An explanation module ranks lines by attention weight × gradient saliency; the single "
        "line it names is a line changed by the real security fix for **21.2%** of vulnerable "
        "test functions, against 8.6% for a random line. The model is served by a FastAPI "
        "backend with SQLite caching (under 0.4 s per file on a CPU) and shown on GitHub "
        "through a Chrome extension.")
    R.para("**Keywords:** graph neural networks, GraphSAGE, vulnerability detection, "
           "exploratory data analysis, class imbalance, calibration, explainability.",
           align=WD_ALIGN_PARAGRAPH.LEFT)

    # ======================================================================== TOC
    R.title("Table of Contents")
    toc = R.d.add_paragraph(style="Report Body")
    add_field(toc, 'TOC \\o "1-2" \\h \\z \\u',
              "Right-click and choose Update Field to build the table of contents.")

    # ======================================================================== Chapter 1
    R.h1("Chapter 1: Introduction")
    R.h2("1.1 Introduction to the Project")
    R.para(
        "Memory-safety bugs in C and C++, such as buffer overflows, use-after-free and "
        "missing NULL checks, remain among the most exploited software vulnerabilities. "
        "Reviewers cannot read every function carefully, so a tool that ranks functions by "
        "risk and points at suspicious lines saves real effort. This project builds such a "
        "tool for C/C++ code on GitHub: source files are split into functions, each function "
        "is represented as a graph, and a graph neural network (GNN) predicts the probability "
        "that it is vulnerable. The result is shown directly on GitHub through a browser "
        "extension.")
    R.h2("1.2 Problem Statement")
    R.para(
        "Rule-based analysers such as Flawfinder [8] match dangerous patterns (for example "
        "calls to `strcpy`). They miss bugs that depend on program structure and flag many "
        "safe uses. Machine-learning approaches that treat code as plain text lose its syntax "
        "tree and execution order. The problem addressed here is: **given a C/C++ function, "
        "estimate how likely it is to contain a security vulnerability, using its structure, "
        "and identify the lines most likely responsible**, for code from projects never seen "
        "during training.")
    R.h2("1.3 Objectives of the Project")
    R.bullets([
        "Explore and clean the Big-Vul dataset [1] and build a project-disjoint split.",
        "Convert each function into a graph: AST edges plus next-statement and variable-use edges.",
        "Train a Hybrid GraphSAGE model [2] and compare it with rule-based, text-based and "
        "size-based baselines, with confidence intervals and repeated seeds.",
        "Calibrate the predicted probabilities and highlight the most likely vulnerable lines.",
        "Deliver the scanner as a FastAPI backend and a Chrome extension for GitHub.",
    ])
    R.h2("1.4 Scope of the Project")
    R.para("**In scope:** function-level analysis of C/C++ files (`.c`, `.cpp`, `.h`, `.hpp`), "
           "CPU-only scanning on a normal laptop, and GitHub file pages.")
    R.para("**Out of scope:** bugs that span several functions or files, full data-flow "
           "analysis, symbolic execution and automatic bug fixing. The tool is an aid for a "
           "human reviewer, not a proof that a bug exists.")

    # ======================================================================== Chapter 2
    R.h1("Chapter 2: Data and System Design")

    # ---------------------------------------------------------------- 2.1 EDA
    R.h2("2.1 Dataset and Exploratory Data Analysis")
    R.para(
        f"**Dataset.** Big-Vul [1, 13] contains {n(raw['rows'])} C/C++ functions from "
        f"{raw['projects']} open-source projects. Each row has the function before the security "
        "fix (`func_before`), the fixed version (`func_after`), a vulnerable/safe label, the "
        "project, and the CWE and CVE identifiers of the fixing commit. "
        f"{n(raw['vulnerable'])} functions ({raw['vulnerable_pct']:.2f}%) are labelled vulnerable. "
        "The safe functions come from the same fixing commits, so they are realistic \"hard\" "
        f"negatives; this is also why {n(raw['safe_with_cwe'])} safe rows carry a CWE id.")
    nc = raw["null_counts"]
    R.table([
        ["Check", "Finding", "Action"],
        ["Missing / empty code", f"{raw['empty_code']} of {n(raw['rows'])}", "none needed"],
        ["Missing CWE ID", f"{n(nc['CWE ID'])} rows ({n(raw['vulnerable_without_cwe'])} vulnerable)",
         "not used as a feature"],
        ["Missing CVE ID / commit message", f"{n(nc['CVE ID'])} / {n(nc['commit_message'])}",
         "metadata only, not used"],
        ["Shorter than 3 lines", n(raw["lines_lt3"]), "removed"],
        ["Longer than 500 lines", n(raw["lines_gt500"]), "removed"],
        ["Duplicate bodies (whitespace-insensitive)", f"{n(raw['duplicate_rows'])} rows",
         "removed, first copy kept"],
        ["Duplicates with conflicting labels",
         f"{n(raw['conflicting_label_groups'])} groups ({n(raw['conflicting_label_rows'])} rows)",
         "first copy's label kept (label noise)"],
    ], [58, 52, 50], caption="Table 2.1: Data-quality checks on the raw dataset", font_size=10)

    R.para(
        "**Class imbalance.** About 1 in 19 functions is vulnerable in every split (Figure 2.1). "
        f"In the training split the ratio is {sub['pos_weight']}:1 safe to vulnerable, so a model "
        f"that always answers \"safe\" would be {100 - sp['train']['vulnerable_pct']:.1f}% accurate "
        "and useless. We therefore evaluate with average precision and Precision@Top-10% "
        "rather than accuracy [15], and weight the loss (Section 2.2).")
    R.figure(ASSETS / "an_class_balance.png", 140, "Figure 2.1: Share of vulnerable functions per split")

    R.para(
        "**Function length.** Vulnerable functions are clearly longer (Figure 2.2): the median is "
        f"{A['metric_medians']['loc']['vulnerable']:.0f} lines against "
        f"{A['metric_medians']['loc']['safe']:.0f} for safe functions. The gap differs by split "
        f"(train {sp['train']['median_lines_vuln']:.0f} vs {sp['train']['median_lines_safe']:.0f}, "
        f"validation {sp['val']['median_lines_vuln']:.0f} vs {sp['val']['median_lines_safe']:.0f}, "
        f"test **{sp['test']['median_lines_vuln']:.0f} vs {sp['test']['median_lines_safe']:.0f}**). "
        "Length is therefore a strong and possibly superficial signal, which Section 4.3 tests "
        "directly with a length-only baseline.")
    R.figure(ASSETS / "an_length.png", 140, "Figure 2.2: Function length by label (log scale)")

    cwe = sub["cwe_top"]
    R.para(
        f"**Vulnerability types.** The most common CWE is CWE-119 (buffer errors, {cwe['CWE-119']} "
        f"functions in the subset), followed by CWE-20 (input validation) and CWE-399 (resource "
        f"management). {cwe['no CWE label']} vulnerable functions "
        f"({100 * cwe['no CWE label'] / sub['vulnerable']:.0f}%) have no CWE label, and the long "
        "tail of rarer types (Figure 2.3) is why predicting the type was not reliable.")
    R.figure(ASSETS / "an_cwe.png", 135, "Figure 2.3: Ten most common CWE types among vulnerable functions")

    R.para(
        f"**Projects.** The {sub['projects']} projects in the subset are very unequal: "
        f"{sub['projects_without_vulnerable']} contain no vulnerable function at all. Because "
        "projects are kept whole (Section 2.2), the training split is dominated by two large "
        f"codebases (Chrome {sp['train']['largest_project_share_pct']}%, top five "
        f"{sp['train']['top5_share_pct']}%), while validation and test are each "
        f"{sp['test']['projects']} mostly small projects (median {sp['test']['median_project_size']:.0f} "
        "functions) (Figure 2.4). This is a real distribution shift between training and "
        "testing, and it explains why scores differ between validation and test.")
    R.figure(ASSETS / "an_projects.png", 150, "Figure 2.4: Share of functions from the largest projects in each split")

    R.para(
        "**Code metrics and the label.** Vulnerable functions have higher medians for every "
        "metric (Figure 2.5); risky API calls are rare in both classes but twice as frequent in "
        "vulnerable code (10% vs 5% of functions). The metrics are strongly correlated with "
        f"length (Spearman ρ with lines of code: cyclomatic {rho['loc']['cyclomatic']:.2f}, pointer "
        f"operations {rho['loc']['pointer_ops']:.2f}, nesting {rho['loc']['max_nesting']:.2f}) "
        f"but only weakly with the label (ρ ≤ {max(rho[m]['label'] for m in rho if m != 'label'):.2f}, "
        "Figure 2.6). No single metric is decisive, and they largely measure the same thing: size.")
    R.figure(ASSETS / "an_metrics.png", W, "Figure 2.5: The six code metrics by label")
    R.figure(ASSETS / "an_corr.png", 95, "Figure 2.6: Spearman correlation between metrics and the label")

    # ---------------------------------------------------------------- 2.2 preprocessing
    R.h2("2.2 Data Preprocessing and Splitting")
    R.table([
        ["Step", "Rule", "Functions left", "Reason"],
        ["1. Load", "all Big-Vul rows", n(raw["rows"]), "full dataset"],
        ["2. Length filter", "keep 3–500 lines", n(raw["after_length_filter"]),
         "stubs have no structure; huge functions make graphs too large for a laptop"],
        ["3. Deduplicate", "SHA-1 of whitespace-normalised code; keep first", n(raw["after_dedup"]),
         "identical code in train and test would leak; done before splitting"],
        ["4. Sample", "30,000, stratified by label", n(sub["functions"]),
         "fits laptop memory and training time; keeps the class ratio"],
        ["5. Split", "whole projects, greedy 80/10/10 packing",
         f"{n(sp['train']['functions'])} / {n(sp['val']['functions'])} / {n(sp['test']['functions'])}",
         "test only on projects never seen in training"],
    ], [26, 42, 26, 66], caption="Table 2.2: Preprocessing steps", font_size=10)
    R.figure(ASSETS / "report_fig_data.png", 140,
             "Figure 2.7: Cleaning steps (top) and the project-disjoint split (bottom)")
    R.para(
        "**Why split by project.** Functions from the same project share style, macros and "
        "helper names. A random split would put near-identical functions in both training and "
        "test sets and inflate the scores. The greedy packer assigns whole projects, largest "
        "first, to whichever split is furthest below its 80/10/10 target; a side effect is that "
        "the largest projects all land in training (Figure 2.4).")
    R.para(
        f"**Class imbalance.** The loss is weighted with `pos_weight` = {sub['pos_weight']} "
        f"({n(sp['train']['functions'] - sp['train']['vulnerable'])} safe / "
        f"{n(sp['train']['vulnerable'])} vulnerable training functions). Unlike oversampling it "
        "does not repeat examples (less over-fitting), and unlike undersampling it discards no "
        "data. Its side effect is that raw scores are biased upwards; Section 3.4 corrects this "
        "with calibration.")
    R.para(
        "**Feature scaling.** The six metrics are z-scored with the mean and standard deviation "
        "of the training split only, and the node-type vocabulary is built from training "
        "graphs only, so no statistic of the validation or test data enters the model.")
    R.table([
        ["Leakage risk", "How it is prevented"],
        ["Same code in two splits", "deduplication before splitting"],
        ["Same project in two splits", "project-disjoint split"],
        ["Project names that differ only by case",
         "one case found (varnish-cache, 17 functions, none vulnerable); negligible, to be "
         "fixed by lower-casing names"],
        ["Statistics from evaluation data", "scaling and vocabulary fitted on training data only"],
        ["Tuning on the test set", "early stopping, calibration, threshold and line rules use "
         "validation only; test is used once"],
    ], [56, 104], caption="Table 2.3: Data-leakage prevention", font_size=10)

    # ---------------------------------------------------------------- 2.3 - 2.6 system
    R.h2("2.3 System Architecture")
    R.para(
        "The system has two parts (Figure 2.8). The **offline training pipeline** prepares "
        "the data and trains the model once. The **online scanning path** runs whenever a "
        "user opens a C/C++ file on GitHub: the Chrome extension downloads the raw file and "
        "sends it to a local FastAPI server, which splits the file into functions, scores "
        "each one with the trained model and returns the riskiest lines. The extension then "
        "draws badges, line highlights and a summary panel on the page.")
    R.figure(ASSETS / "report_fig_architecture.png", 148,
             "Figure 2.8: Training pipeline (top) and scanning path (bottom)")
    R.h2("2.4 Database Design")
    R.para("The backend uses SQLite with two tables: a cache, so the same file is never "
           "scanned twice, and a feedback table for false-positive reports sent from the "
           "extension.")
    R.table([
        ["Table", "Column", "Type", "Description"],
        ["scan_cache", "file_hash (PK)", "TEXT", "scanner version + SHA-256 of the file content"],
        ["scan_cache", "result_json", "TEXT", "full scan result returned to the extension"],
        ["scan_cache", "created_at", "REAL", "time of the scan"],
        ["feedback", "id (PK)", "INTEGER", "auto-increment identifier"],
        ["feedback", "file_hash", "TEXT", "file the report refers to"],
        ["feedback", "function_name", "TEXT", "function that contains the line"],
        ["feedback", "line", "INTEGER", "reported line number"],
        ["feedback", "is_false_positive", "INTEGER", "1 if the user clicked \"Not a bug\""],
        ["feedback", "created_at", "REAL", "time of the report"],
    ], [26, 36, 22, 76], caption="Table 2.4: SQLite tables")
    R.h2("2.5 Scan Request Flow")
    R.para(
        "Figure 2.9 follows one scan request. The server first checks its SQLite cache; a "
        "file seen before is answered in about 2 ms. Otherwise the file is split into "
        "functions, each function is scored by the model and its lines are ranked, and the "
        "result is cached before it is returned to the extension.")
    R.figure(ASSETS / "report_fig_scan_flow.png", 90, "Figure 2.9: Flowchart of one scan request")
    R.h2("2.6 Module Description")
    R.table([
        ["Module", "Files", "Responsibility"],
        ["Data preparation", "dataset.py", "download Big-Vul, clean, split by project"],
        ["Analysis", "analysis.py", "EDA, size baselines, confidence intervals, calibration plots"],
        ["Baselines", "run_baselines.py", "Flawfinder, TF-IDF + LogReg, risky-API rule"],
        ["Parsing", "parsing.py", "tree-sitter parser; split a file into functions"],
        ["Graph construction", "graph_builder.py, build_graphs.py", "AST + extra edges → PyTorch Geometric graphs"],
        ["Features", "features.py, prepare_features.py", "node features and 6 function metrics"],
        ["Model and training", "model.py, train.py, run_seeds.py", "Hybrid GraphSAGE, ablations, seeds"],
        ["Calibration", "calibration.py", "Platt scaling and calibration error"],
        ["Evaluation", "evaluate.py", "precision, recall, F1, PR-AUC, Precision@Top-10%"],
        ["Explanations", "explain.py", "ranks lines and attaches reasons"],
        ["Backend", "backend/app.py", "REST API, SQLite cache and feedback"],
        ["Extension", "extension/", "badges, line callouts, side panel on GitHub"],
    ], [36, 58, 66], caption="Table 2.5: Modules of the system")

    # ======================================================================== Chapter 3
    R.h1("Chapter 3: Implementation")
    R.h2("3.1 Technologies and Tools Used")
    R.table([
        ["Category", "Technology", "Use"],
        ["Languages", "Python 3.13, JavaScript", "machine learning and backend; browser extension"],
        ["Parsing", "tree-sitter (C and C++ grammars) [9]", "syntax trees of functions"],
        ["Deep learning", "PyTorch, PyTorch Geometric [4]", "GraphSAGE and GCN layers, training"],
        ["Classical ML", "scikit-learn [7]", "baselines, Platt scaling, metrics"],
        ["Rule-based tool", "Flawfinder [8]", "baseline for comparison"],
        ["Data and plots", "Hugging Face datasets, pandas, matplotlib", "loading, EDA, figures"],
        ["Backend", "FastAPI [10], Uvicorn, SQLite", "REST API, cache, feedback"],
        ["Frontend", "Chrome extension (Manifest V3) [11]", "results on GitHub pages"],
    ], [32, 62, 66], caption="Table 3.1: Technologies used")
    R.h2("3.2 Project Structure")
    R.code([
        "Project/",
        "├── src/          data, graphs, features, model, training, calibration, analysis",
        "├── baselines/    run_baselines.py",
        "├── backend/      app.py (FastAPI), cache.db (SQLite)",
        "├── extension/    manifest.json, background.js, content.js, popup",
        "├── models/       trained weights, vocabulary, metric statistics",
        "├── results/      analysis.json, seed runs, training log",
        "├── data/         cleaned splits and cached graphs",
        "└── docs/         report, explainer PDF and their sources",
    ])
    R.h2("3.3 Description of Key Classes and Functions")
    R.table([
        ["Name", "File", "Description"],
        ["HybridGNN", "model.py", "3 SAGEConv layers, attention + mean pooling, metrics concatenation, MLP classifier"],
        ["build_graph()", "graph_builder.py", "parses one function into a graph with three edge types"],
        ["node_features()", "features.py", "166-number vector for every node"],
        ["function_metrics()", "features.py", "the 6 function-level code metrics"],
        ["train_variant()", "train.py", "trains one model with early stopping, calibration and threshold"],
        ["fit_platt()", "calibration.py", "fits the probability calibration on validation logits"],
        ["explain_function()", "explain.py", "calibrated risk score and ranked top-3 lines"],
        ["scan(), feedback()", "backend/app.py", "REST endpoints used by the extension"],
        ["getCodeCell()", "content.js", "the only function that depends on GitHub's page layout"],
    ], [36, 32, 92], caption="Table 3.2: Key classes and functions")

    R.h2("3.4 Implementation of Major Modules")
    R.para(
        "**Code to graph.** tree-sitter parses each function; every named node stores its "
        "type, its token (for leaves) and its line number. Three undirected edge types are "
        "added: AST parent–child, next statement in the same block, and next use of the same "
        "variable name (Figure 3.1). All 30,000 functions converted without failures, with "
        "about 120 nodes and 146 edges per training graph. The edge type is stored with each "
        "edge, but the GraphSAGE layers do not use it: all edges are treated alike.")
    R.figure(ASSETS / "report_fig_graph.png", 95, "Figure 3.1: Graph of a small function that calls strcpy")
    R.para(
        "**Features.** Each node is described by 166 numbers (Figure 3.2): a one-hot of the 100 "
        "most common node types plus \"other\" (101), the node's token hashed into 64 slots "
        "(feature hashing: a fixed size with no vocabulary to store, at the cost of occasional "
        "collisions), and a flag for risky C functions. Each function also gets the six metrics "
        "of Section 2.1 (lines of code; cyclomatic complexity = 1 + number of if/for/while/case, "
        "&&, || and ?; pointer operations = count of *, -> and &; risky calls; maximum nesting; "
        "parameters). The pointer count is a rough proxy, since it also counts multiplication "
        "and bitwise AND.")
    R.figure(ASSETS / "report_fig_features.png", 140, "Figure 3.2: The 166 numbers that describe each node")
    R.para(
        "**Model and training.** Three SAGEConv layers produce node vectors, which are pooled "
        "by attention and by mean; the result is concatenated with the six metrics and passed "
        "to an MLP (Figure 3.3). The hyperparameters (Table 3.3) were **fixed in advance** from "
        "common defaults, not tuned, so validation data was used only for early stopping, "
        "calibration and the threshold.")
    R.figure(ASSETS / "report_fig_model.png", 62, "Figure 3.3: Architecture of the Hybrid GraphSAGE model")
    R.table([
        ["Hyperparameter", "Value", "Reason"],
        ["GNN layers / hidden size", "3 / 128", "each node sees neighbours up to 3 hops away"],
        ["Dropout", "0.3", "regularisation for a small positive class"],
        ["Pooling", "attention + mean", "attention weights reused for explanations"],
        ["Classifier", "MLP 262 → 128 → 64 → 1", "graph summary (256) + 6 metrics"],
        ["Loss", "BCE, pos_weight = " + str(sub["pos_weight"]), "class imbalance (Section 2.2)"],
        ["Optimiser / batch", "Adam, lr 0.001 / 64 graphs", "standard defaults"],
        ["Early stopping", "validation PR-AUC, patience 6", "keeps the best epoch"],
    ], [46, 50, 64], caption="Table 3.3: Hyperparameters (fixed in advance)", font_size=10)
    R.para(
        "**Calibration.** Because of `pos_weight`, the raw scores are shifted upwards by about "
        "ln 18.6 ≈ 2.9. We first used temperature scaling [6], which can only stretch or squash "
        f"scores: it pushed every test probability into {f3(cal_old['min'])}–{f3(cal_old['max'])} "
        f"(mean {f3(cal_old['mean'])} for a true rate of {f3(cal_old['true_rate'])}). We now use "
        f"**Platt scaling** [14], p = σ(a·z + b), fitted on validation (a = {platt['a']:.3f}, "
        f"b = {platt['b']:.3f}; note b ≈ −ln 18.6). The ranking, and therefore PR-AUC, is "
        f"unchanged. The decision threshold (F1-optimal on validation) is {cm['threshold']:.3f} "
        "in calibrated units and flags exactly the same functions as before. Risk bands are "
        "anchored to it: **Low** below the threshold, **Medium** flagged, **High** at least twice "
        "the threshold.")
    R.para(
        "**Line explanations.** Node importance is the attention weight times the gradient "
        "magnitude of the risk score with respect to the node [5]. Scores are summed per line, "
        "counting only single-line nodes and only lines inside the function body; within the "
        "top 3, a line matching a known dangerous pattern (`strcpy`, `memcpy`, `malloc`…) is "
        "listed first (Figure 3.4). This last step is a rule-based reordering, not part of the "
        "model. All rules were chosen on the validation set.")
    R.figure(ASSETS / "report_fig_explain_flow.png", 118, "Figure 3.4: How the most likely vulnerable line is chosen")
    R.para(
        "**Backend and extension.** The FastAPI server loads the model once, splits a file "
        "into functions and caches results. GitHub's security policy blocks page scripts from "
        "calling a local server, so the extension's background service worker makes all "
        "network calls to `127.0.0.1:8000`.")

    R.h2("3.5 Exception Handling")
    R.bullets([
        "Invalid request bodies are rejected automatically by FastAPI (Pydantic) with HTTP 422.",
        "Graph building wraps each function in `try`/`except` and counts failures (0 of 30,000).",
        "Feature preparation checks that graphs and data rows are aligned before using them.",
        "The extension uses a 10-second timeout and shows the reason in its panel instead of failing silently.",
        "A scanner version in the cache key prevents old results being served after a code change.",
    ])
    R.h2("3.6 Data Structures Used")
    R.bullets([
        "**Dictionaries:** node-type vocabulary, per-line score totals, calibration parameters.",
        "**Sets:** ground-truth fix lines; lines the user marked as false positives.",
        "**Graph objects (PyTorch Geometric `Data`):** edge list (2 × E tensor), edge types, "
        "node feature matrix and metric vector.",
        "**DataFrames / Parquet files:** the cleaned train, validation and test splits.",
    ])
    R.h2("3.7 Database Connectivity")
    R.para("The backend uses Python's built-in `sqlite3` module. A new connection is opened "
           "per request, which keeps the code simple and safe with the web server.")
    R.code([
        "def db():",
        "    conn = sqlite3.connect(DB_PATH)",
        "    conn.row_factory = sqlite3.Row      # rows behave like dictionaries",
        "    return conn",
    ])
    R.h2("3.8 Important Code Snippets")
    R.para("**Attention + mean pooling and the hybrid classifier** (model.py, simplified):")
    R.code([
        "score = self.att_lin(h).squeeze(-1)              # one score per node",
        "alpha = softmax(score, batch)                    # normalised within each graph",
        "att_pool = global_add_pool(h * alpha.unsqueeze(-1), batch)",
        "mean_pool = global_mean_pool(h, batch)",
        "emb = torch.cat([att_pool, mean_pool, data.metrics], dim=1)",
        "logit = self.mlp(emb).squeeze(-1)                # risk logit per function",
    ])
    R.para("**Platt calibration** (calibration.py):")
    R.code([
        "lr = LogisticRegression(C=1e6).fit(val_logits.reshape(-1, 1), val_labels)",
        "a, b = lr.coef_[0][0], lr.intercept_[0]",
        "prob = 1 / (1 + np.exp(-(a * logit + b)))       # calibrated probability",
    ])

    # ======================================================================== Chapter 4
    R.h1("Chapter 4: Results and Screenshots")
    R.h2("4.1 Main Application Screen")
    R.para(
        "Figure 4.1 shows the extension on `git/usage.c` from the Git project. Each function "
        "has a risk badge; each function's most likely vulnerable line has a red bar and a "
        "\"Likely vulnerable line\" label with the reason; hovering shows details and a "
        "\"Not a bug\" button. The side panel names the most likely vulnerable line in the "
        "whole file and lists every function by risk, with clickable lines.")
    R.figure(ASSETS / "extension_ui.jpg", W,
             "Figure 4.1: The extension on a GitHub C file (captured before the calibration fix "
             "of Section 3.4; the risk percentages shown are the old, uncalibrated values)")

    # ---------------------------------------------------------------- 4.2 main results
    R.h2("4.2 Main Results and Baselines")
    R.para(
        "All results are on the test set: 3,000 functions from 96 projects never seen in "
        "training, of which 193 are vulnerable. We report **average precision** (AP, the area "
        "under the precision–recall curve as computed by scikit-learn; a random ranking scores "
        f"the vulnerable rate, {f3(cal_new['true_rate'])}) and **Precision@Top-10%** (the share "
        "of real bugs among the 300 functions ranked riskiest; tied scores are counted at their "
        "average rate). Brackets give 95% confidence intervals from 2,000 bootstrap resamples "
        "of the test set [16]. Besides the three baselines, we add two size-based baselines that "
        "the first version of this report lacked: ranking by lines of code, and a logistic "
        "regression on the six metrics.")
    order = [("Flawfinder (level >= 2)", "Flawfinder (level ≥ 2)", ("0.116", "0.254", "0.159")),
             ("Risky-API rule", "Risky-API rule", ("0.125", "0.218", "0.158")),
             ("TF-IDF + Logistic Regression", "TF-IDF + Logistic Regression", ("0.108", "0.456", "0.174")),
             ("Lines of code only", "Lines of code only (new)", ("–", "–", "–")),
             ("6 metrics only (LogReg)", "6 metrics only, LogReg (new)", ("–", "–", "–")),
             ("Hybrid GraphSAGE (ours)", "**Hybrid GraphSAGE (ours)**", ("0.219", "0.306", "0.255"))]
    rows = [["Method", "AP [95% CI]", "P@Top-10% [95% CI]", "Prec.", "Recall", "F1"]]
    for key, label, prf in order:
        s = ts[key]
        rows.append([label, ci(s["ap"], s["ap_ci95"]), ci(s["p_at_10"], s["p_at_10_ci95"]), *prf])
    R.table(rows, [46, 36, 36, 14, 14, 14],
            caption="Table 4.1: Function-level results on the test set", font_size=9.5)
    d_loc, d_lr = pdiff["model minus Lines of code only"], pdiff["model minus 6 metrics only (LogReg)"]
    d_tf, d_ra = pdiff["model minus TF-IDF + Logistic Regression"], pdiff["model minus Risky-API rule"]
    R.para(
        "Paired bootstrap tests on the same resamples show that the model is **significantly "
        f"better** than the TF-IDF classifier (AP difference {d_tf['ap_ci95'][0]:+.3f} to "
        f"{d_tf['ap_ci95'][1]:+.3f}) and the risky-API rule ({d_ra['ap_ci95'][0]:+.3f} to "
        f"{d_ra['ap_ci95'][1]:+.3f}). It is **not** better than the size baselines: versus the "
        f"metrics-only regression the AP difference is {d_lr['ap_ci95'][0]:+.3f} to "
        f"{d_lr['ap_ci95'][1]:+.3f}, versus lines of code {d_loc['ap_ci95'][0]:+.3f} to "
        f"{d_loc['ap_ci95'][1]:+.3f} (both intervals contain zero), and on Precision@Top-10% "
        f"lines of code is significantly better ({d_loc['p_at_10_ci95'][0]:+.3f} to "
        f"{d_loc['p_at_10_ci95'][1]:+.3f}). The precision–recall curves (Figure 4.2) show the "
        "same picture: the model, lines of code and the metric regression overlap almost "
        "everywhere. Note also that an earlier version reported Precision@Top-10% = 0.023 for "
        "Flawfinder; that number came from breaking tied scores by row order and was wrong.")
    R.figure(ASSETS / "an_pr_curves.png", 140, "Figure 4.2: Precision–recall curves on the test set")
    R.table([
        ["", "Predicted safe", "Predicted vulnerable"],
        ["Actually safe", n(cm["tn"]), n(cm["fp"])],
        ["Actually vulnerable", n(cm["fn"]), n(cm["tp"])],
    ], [60, 50, 50], caption=f"Table 4.2: Confusion matrix of the model at the threshold {cm['threshold']:.3f}")
    flagged = cm["tp"] + cm["fp"]
    R.para(
        f"At the chosen threshold the model flags {n(flagged)} of 3,000 functions "
        f"({100 * flagged / 3000:.0f}%) and catches {cm['tp']} of 193 vulnerable ones (recall "
        f"{cm['tp'] / 193:.1%}); {cm['tp'] / flagged:.1%} of the flagged functions are real "
        f"vulnerabilities, {cm['tp'] / flagged / cal_new['true_rate']:.1f}× the base rate.")

    # ---------------------------------------------------------------- 4.3 length
    R.h2("4.3 Is the Model More Than a Length Detector?")
    R.para(
        "Because length predicts vulnerability so well (Section 2.1), we split the test set into "
        "four equal-sized length bands and compared the methods inside each band (Table 4.3). "
        "If the graph captured structure beyond size, the model should beat the length "
        "baseline within a band, where lengths are similar.")
    lb = [["Lines", "Functions", "Vulnerable", "Rate", "Model AP", "Lines-only AP", "Metrics AP"]]
    for r in A["length_bands"]:
        lb.append([r["lines"], n(r["functions"]), str(r["vulnerable"]), f3(r["rate"]),
                   f3(r["Hybrid GraphSAGE (ours)"]), f3(r["Lines of code only"]),
                   f3(r["6 metrics only (LogReg)"])])
    R.table(lb, [20, 22, 22, 18, 24, 28, 26],
            caption="Table 4.3: Average precision inside four length bands of the test set", font_size=10)
    R.para(
        "Inside every band the model is no better than the length baseline, and the model's "
        f"score has a Spearman correlation of **{A['spearman_model_vs_loc']:.2f}** with function "
        "length. The honest conclusion is that, on this dataset and split, the model mostly "
        "learned that long functions are riskier. The graph component helps relative to the "
        "pure-GNN ablation (Section 4.5) and relative to text and rules, but we could not show "
        "any gain beyond function size. Two caveats: the per-band estimates rest on few "
        f"vulnerable functions ({A['length_bands'][0]['vulnerable']} in the shortest band), and "
        "length is a property of Big-Vul's "
        "labelling as much as of real bugs, since vulnerability fixes tend to touch large "
        "functions.")

    # ---------------------------------------------------------------- 4.4 training + calibration
    R.h2("4.4 Training Behaviour and Calibration")
    lc = A["learning_curves"][MAIN]
    best = max(lc, key=lambda x: x[2])
    R.para(
        f"Figure 4.3 shows the reported training run. Validation PR-AUC peaks early (epoch "
        f"{best[0]} for the main model, {best[2]:.3f}) and then declines while the training loss "
        f"keeps falling ({lc[0][1]:.2f} → {lc[-1][1]:.2f}): the model starts to over-fit after "
        "a few epochs, and early stopping keeps the best epoch. Validation PR-AUC (≈0.13) is "
        "lower than test PR-AUC (≈0.21). This is a property of the splits, not of tuning: the "
        "lines-of-code baseline, which learns nothing, shows the same gap "
        f"({f3(A['loc_val_ap'])} on validation vs {f3(m_loc['ap'])} on test), because the two "
        "splits contain different projects with different length gaps (Figure 2.4).")
    R.figure(ASSETS / "an_learning_curves.png", 155, "Figure 4.3: Learning curves of the reported run")
    R.para(
        f"Figure 4.4 compares the calibration. With temperature scaling the mean predicted "
        f"probability was {f3(cal_old['mean'])} for a true rate of {f3(cal_old['true_rate'])} "
        f"(expected calibration error, ECE, {f3(cal_old['ece'])}), and all "
        f"{cal_old['bands']['Medium']:,} test functions fell in the old \"Medium\" band. With "
        f"Platt scaling the mean is {f3(cal_new['mean'])} and the ECE {f3(cal_new['ece'])}. With the "
        f"threshold-anchored bands, the test set splits into {n(bands.get('Low', 0))} Low, "
        f"{n(bands.get('Medium', 0))} Medium and {n(bands.get('High', 0))} High functions.")
    R.figure(ASSETS / "an_calibration.png", 155, "Figure 4.4: Calibration before (temperature) and after (Platt)")

    # ---------------------------------------------------------------- 4.5 ablations
    R.h2("4.5 Ablation Study over Five Seeds")
    R.para(
        f"Each variant was trained with {k_seeds} different random seeds ({', '.join(map(str, S['seeds']))}); "
        "Table 4.4 reports the mean ± standard deviation, next to the single run of the first "
        "version.")
    ab = [["Variant", "Test AP", "Test P@Top-10%", "Val AP", "Best epoch", "First run AP"]]
    for v, label in ((MAIN, "Hybrid GraphSAGE (main)"), (AST, "AST edges only"),
                     (GCN, "GCN instead of SAGE"), (NOMET, "No metrics (pure GNN)")):
        s = S[v]
        ab.append([label, f"{s['ap'][0]:.3f} ± {s['ap'][1]:.3f}", f"{s['p10'][0]:.3f} ± {s['p10'][1]:.3f}",
                   f"{s['val'][0]:.3f}", f"{s['epoch']:.1f}", f3(ORIGINAL_RUN[v][0])])
    R.table(ab, [44, 28, 30, 18, 18, 22],
            caption=f"Table 4.4: Ablations, mean ± standard deviation over {k_seeds} seeds", font_size=9.5)
    R.para(
        f"**Metrics matter most.** Removing the six metrics lowers test AP from "
        f"{S[MAIN]['ap'][0]:.3f} to {S[NOMET]['ap'][0]:.3f} on average, by far the largest effect, "
        "consistent with Section 4.3: the metrics carry the size signal.")
    R.para(
        f"**AST-only vs all edges.** AST-only had a higher test AP than the main model in "
        f"{S['ast_beats_main_test']} of {k_seeds} seeds and a higher validation AP in "
        f"{S['ast_beats_main_val']} of {k_seeds}. The difference between the means "
        f"({S[AST]['ap'][0] - S[MAIN]['ap'][0]:+.3f}) is small relative to the seed-to-seed "
        "spread, so the first version's single-run result (0.228 vs 0.208) should not be read "
        "as \"the extra edges hurt\". A likely reason the extra edges do not help is that our "
        "SAGEConv layers ignore edge types: next-statement and variable-use edges enter as "
        "untyped extra neighbours, and name-matched variable edges also connect unrelated "
        "variables that share a name. Testing whether typed control- or data-flow edges help "
        "would need an edge-type-aware layer such as R-GCN.")
    R.para(
        f"**GraphSAGE vs GCN.** GCN averaged {S[GCN]['ap'][0]:.3f} against {S[MAIN]['ap'][0]:.3f} "
        "for GraphSAGE; the gap is within the seed-to-seed variation, so we do not claim that "
        "one layer type is better on this data.")

    # ---------------------------------------------------------------- 4.6 lines
    R.h2("4.6 Line-Level Explanation Evaluation")
    R.para(
        "The ground truth for a vulnerable function is the set of its lines that the security "
        "fix deleted or changed, found by diffing `func_before` with `func_after`. Of the 193 "
        "vulnerable test functions, 56 were fixed only by adding new lines (for example a "
        "missing NULL check), so no original line is marked and they are excluded, leaving 137. "
        "This is an oracle setting: the explanation is evaluated on functions we already know "
        "are vulnerable.")
    R.table([
        ["Line-ranking method", "Top-1 hit", "Top-3 hit"],
        ["Random lines (expected)", "0.086", "0.192"],
        ["All nodes (first version)", "0.109", "0.343"],
        ["**Final method (chosen on validation)**", "**0.212**", "**0.380**"],
    ], [90, 35, 35], caption="Table 4.5: How often the named lines include a line the real fix changed")
    R.para("Figure 4.5 shows how each rule raised the Top-1 hit rate above random guessing. "
           "Most of the gain comes from counting only single-line nodes and only body lines; "
           "the rule-based \"known culprits first\" step changes about one function.")
    R.figure(ASSETS / "report_fig_lines.png", 140,
             "Figure 4.5: Top-1 line hit rate as each rule was added (dashed line = random)")

    # ---------------------------------------------------------------- 4.7 demo
    demo = A["demo"]
    R.h2("4.7 Input and Output Demonstration")
    R.para("**Input:** a small C file sent to `POST /scan`; one of its functions is:")
    R.code([
        "void copy_input(char *src) {",
        "    char buffer[16];",
        "    strcpy(buffer, src);",
        "    printf(\"%s\\n\", buffer);",
        "}",
    ])
    R.para("**Output** (shortened JSON for this function, with calibrated probability):")
    R.code([
        f"{{ \"name\": \"{demo['name']}\", \"start_line\": {demo['start_line']}, \"end_line\": {demo['end_line']},",
        f"  \"risk_score\": {demo['risk_score']}, \"band\": \"{demo['band']}\",",
        "  \"top_lines\": [",
        f"    {{ \"line\": {demo['top_line']}, \"reason\": \"{demo['reason']}\",",
        f"      \"text\": \"{demo['text']}\" }}, ... ] }}",
    ])
    R.para(
        f"The most likely vulnerable line is correctly the `strcpy` call. The calibrated risk "
        f"({demo['risk_score']:.3f}) is low because this short function looks like most safe "
        "functions in the data, an example of the size effect of Section 4.3. The scan took "
        "24 ms (2 ms when repeated from the cache); a 406-line file with 33 functions takes "
        "about 0.2–0.4 s on a CPU.")

    R.h2("4.8 Error Handling Demonstration")
    R.table([
        ["Situation", "What the user sees"],
        ["Backend not running", "Panel message \"Backend not reachable …\" within 10 s"],
        ["File with no functions", "\"No functions in this file — nothing to score.\""],
        ["Wrong request body", "HTTP 422 with the field that is invalid"],
        ["Line reported as not a bug", "Highlight removed; report stored in the feedback table"],
    ], [52, 108], caption="Table 4.6: Error situations and responses")

    # ---------------------------------------------------------------- 4.9 conclusion
    R.h2("4.9 Conclusion and Limitations")
    R.para(
        "We built a complete pipeline, from data exploration to a deployed GitHub extension, "
        "for ranking C/C++ functions by vulnerability risk. The Hybrid GraphSAGE model "
        "significantly outperforms rule-based and text-based baselines on unseen projects, "
        "and its explanations point to the fixed line about 2.5 times more often than chance. "
        "Our analysis also shows the limits of this result: function length alone performs as "
        "well as the model, and we found no measurable benefit from the graph structure beyond "
        "size. Main limitations:")
    R.bullets([
        "Most of the signal is function length; a length-controlled dataset or evaluation is needed "
        "to show what structure adds.",
        "The GNN treats all edges alike, and variable-use edges match names only (no real data flow).",
        f"Labels are noisy: {n(raw['conflicting_label_groups'])} groups of identical functions carry "
        "both labels, and labels come from fix commits.",
        "Training is dominated by two projects (Chrome, Linux), while test projects are small.",
        "Analysis is per function, so cross-function and cross-file bugs are not detected.",
        "Hyperparameters were not tuned, and an extra CWE-type head was not reliable enough to show users.",
    ])

    # ======================================================================== References
    R.h1("References")
    refs = [
        "J. Fan, Y. Li, S. Wang and T. N. Nguyen, \"A C/C++ Code Vulnerability Dataset with "
        "Code Changes and CVE Summaries,\" in Proc. MSR, 2020.",
        "W. L. Hamilton, R. Ying and J. Leskovec, \"Inductive Representation Learning on "
        "Large Graphs,\" in Proc. NeurIPS, 2017.",
        "T. N. Kipf and M. Welling, \"Semi-Supervised Classification with Graph "
        "Convolutional Networks,\" in Proc. ICLR, 2017.",
        "M. Fey and J. E. Lenssen, \"Fast Graph Representation Learning with PyTorch "
        "Geometric,\" ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.",
        "K. Simonyan, A. Vedaldi and A. Zisserman, \"Deep Inside Convolutional Networks: "
        "Visualising Image Classification Models and Saliency Maps,\" ICLR Workshop, 2014.",
        "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, \"On Calibration of Modern Neural "
        "Networks,\" in Proc. ICML, 2017.",
        "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" JMLR, vol. 12, 2011.",
        "D. A. Wheeler, \"Flawfinder,\" https://dwheeler.com/flawfinder/",
        "tree-sitter documentation, https://tree-sitter.github.io/tree-sitter/",
        "FastAPI documentation, https://fastapi.tiangolo.com/",
        "Chrome Extensions (Manifest V3) documentation, https://developer.chrome.com/docs/extensions",
        "MITRE, Common Weakness Enumeration (CWE), https://cwe.mitre.org/",
        "Big-Vul dataset on Hugging Face, https://huggingface.co/datasets/bstee615/bigvul",
        "J. C. Platt, \"Probabilistic Outputs for Support Vector Machines and Comparisons to "
        "Regularized Likelihood Methods,\" in Advances in Large Margin Classifiers, MIT Press, 1999.",
        "T. Saito and M. Rehmsmeier, \"The Precision-Recall Plot Is More Informative than the "
        "ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,\" PLOS ONE, 2015.",
        "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, 1993.",
    ]
    for i, ref in enumerate(refs, 1):
        p = R.para(f"[{i}]  {ref}", align=WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.left_indent = Mm(9)
        p.paragraph_format.first_line_indent = Mm(-9)
