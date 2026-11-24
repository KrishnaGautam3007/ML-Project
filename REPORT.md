# Report material: abstract, viva questions, challenges

All numbers below come from `results/analysis.json` and `results/seed_runs.json`,
the same files the report is built from.

## Abstract

Rule-based static analyzers for C/C++ miss structural bugs and raise many false
alarms, while text-based machine-learning detectors discard the tree structure and
execution order of code. This project presents a vulnerability scanner that
represents each C/C++ function as a graph, built from its tree-sitter Abstract
Syntax Tree plus next-statement and name-matched variable-use edges, and classifies
it with a Hybrid GraphSAGE network trained from scratch. The network combines an
attention-plus-mean-pooled graph embedding with six hand-crafted code metrics.

On a project-disjoint split of the Big-Vul dataset (30,000 functions, 5.3%
vulnerable), the model reaches an average precision (PR-AUC) of 0.208 (95%
bootstrap CI 0.164–0.267; 0.222 ± 0.011 over five seeds). That is significantly
better than a rule-based scanner, a risky-API rule and a TF-IDF text classifier.

A detailed analysis, however, shows that **function length alone reaches 0.223**,
statistically indistinguishable from the model. Inside length bands the model is no
better than length, and its score correlates 0.87 with function size: on this data
most of the learnable signal is size. We also corrected the probability calibration
with Platt scaling (expected calibration error 0.446 → 0.014). In the explanations,
the single top-ranked line is a line changed by the real fix for 21.2% of vulnerable
test functions (8.6% for a random line). The model is served by a FastAPI backend
(under 0.4 s per file on CPU) and surfaced on GitHub through a Manifest V3 Chrome
extension.

---

## Likely viva questions (with short answers)

**1. Why GraphSAGE, and not a plain GCN?**
GraphSAGE updates a node by **aggregating its neighbours** (we use the mean) and
combining that with the node's own vector, using **separate learned weights** for
"self" and "neighbours". So a node keeps its own identity (e.g. "this is a `strcpy`
call") instead of being blurred into its surroundings. It is also **inductive**: it
learns a function that works on graphs never seen in training, which is our case.
GCN averages a node with its neighbours using degree-based normalisation and one
shared weight matrix.

Honest answer on our data: over five seeds the two are **tied** (PR-AUC 0.222 ± 0.011
for SAGE, 0.219 ± 0.011 for GCN). The first single run (0.208 vs 0.194) suggested
otherwise, which is exactly why we repeated it. GraphSAGE was originally proposed
with neighbour *sampling* for huge graphs; our graphs are small, so we aggregate all
neighbours.

**2. Why split the data by project instead of randomly?**
Functions from the same project share coding style, macros, helper functions and
naming. With a random split, near-identical functions land in both train and test,
and the model can score well by memorising a project's style. That is
**information leakage**. Splitting by project means the test set contains only
projects the model never saw, which matches the real use case: scanning a new
GitHub repository.

There is a side effect we found in the EDA: whole projects are assigned
largest-first, so **Chrome (50.1%) and Linux (31.6%) dominate training**, while
validation and test are 96 small projects each. That distribution shift is why
validation PR-AUC (≈0.13) is lower than test (≈0.21). The untrained length baseline
shows the same gap (0.130 vs 0.223), so it isn't caused by tuning.

**3. Why PR-AUC instead of accuracy?**
Only 5.3% of functions are vulnerable. A model that always says "safe" gets about
95% accuracy while finding zero bugs. PR-AUC (here: average precision) only looks at
how well the **vulnerable class** is ranked, across all thresholds, so the big safe
majority cannot inflate it. A random model scores about the positive rate (0.064 on
test). We also report Precision@Top-10%: "if a reviewer checks the 10% riskiest
functions, how many are real bugs?". For rule-based tools with many tied scores we
count the tied group at its average rate, because breaking ties by row order is
arbitrary.

**4. What do the hybrid (hand-crafted) features add?**
They give the classifier global facts that a 3-layer GNN sees only locally: size,
complexity, pointer use, risky calls. Removing them drops PR-AUC from **0.222 to
0.127** (five-seed means), the largest effect of any ablation. But the EDA shows the
metrics mostly measure size: they correlate 0.63–0.86 with lines of code and only
≤ 0.14 with the label. A logistic regression on the 6 metrics alone reaches 0.197.
So what the metrics add is mainly the size signal.

**5. Is your model just a length detector?** *(the hardest question; answer it directly)*
Largely, yes, on this dataset:
- Ranking test functions by lines of code alone gives PR-AUC **0.223**, against
  0.208 for the model. A paired bootstrap shows no significant difference in AP, and
  length is significantly better on P@Top-10%.
- Inside four length bands the model never beats the length baseline.
- The model's score correlates **0.87** with length.

What the GNN *does* show: it is significantly better than text and rule tools, and
the graph part helps compared with a pure GNN. Part of the cause is the dataset:
in Big-Vul, fixes tend to touch long functions, especially in our test projects
(vulnerable median 79 lines vs 22). Showing what structure adds beyond size would
need a length-controlled evaluation. We report this rather than hide it, because
an untested length baseline would make every other number misleading.

**6. How did you handle class imbalance, and why did calibration need fixing?**
- We weight the loss with `pos_weight` = 18.64, the safe-to-vulnerable ratio in
  training. It keeps all data, unlike undersampling, and repeats nothing, unlike
  oversampling.
- The side effect is that raw scores shift up by about ln 18.6 ≈ 2.9.
- Our first calibration, temperature scaling, can only stretch or squash scores, not
  shift them. It squeezed every probability into 0.405–0.643 (mean 0.510 for a true
  rate of 0.064), so every function was "Medium".
- **Platt scaling** σ(a·z + b) adds the missing shift (b = −3.07 ≈ −ln 18.6).
  Calibration error went 0.446 → 0.014.
- Ranking and flagged functions are unchanged, because both mappings are monotonic.
- Risk bands are now anchored to the decision threshold.

**7. How did you prevent data leakage?**
- Deduplicated (whitespace-insensitive SHA-1) **before** splitting.
- Project-disjoint split.
- Feature scaling and the node-type vocabulary fitted on **training data only**.
- Early stopping, calibration, the threshold and the line-ranking rules all use
  **validation only**; the test set is used once.
- One leak found: `varnish-cache` (train) vs `Varnish-Cache` (test), 17 functions,
  none vulnerable, so the effect is negligible. Project names should be lower-cased.

**8. How are the line-level explanations computed, and can you trust them?**
Each node gets **attention weight** (how much the pooling layer relied on it) ×
**gradient saliency** (how much the risk score changes if that node's features
change). Node scores are summed per source line, counting only **single-line nodes**
and only **body lines**. Within the top 3, lines matching a known dangerous pattern
go first; that step is a rule-based reordering, not the model. All rules were chosen
on validation.

On test, the named line is a real fix line 21.2% of the time (8.6% random), and the
top 3 contain one 38.0% of the time (19.2% random). This is evaluated only on
functions known to be vulnerable, and on 137 of 193 (56 fixes only *added* lines).
It shows **where the model looked, not proof of a bug**.

**9. How do you know the differences between your models are real?**
- **95% bootstrap confidence intervals** (2,000 test resamples) for every method,
  and **paired** bootstrap tests between methods.
- **Five seeds** per model variant.

These checks changed our conclusions. The single-run "AST-only beats the full model"
(0.228 vs 0.208) turned out to be noise (0.218 vs 0.222 over five seeds; AST-only won
2 of 5). The test set has only 193 vulnerable functions, so differences below about
0.02 PR-AUC are not meaningful.

**10. What are the main limitations?**
- **Mostly a size signal** (question 5).
- **Edges are untyped in the model** (SAGEConv ignores edge types), and variable-use
  edges match names only, not real data flow.
- **Noisy labels:** fix-commit labels, and 916 groups of identical functions carry
  both labels.
- **Unbalanced projects:** Chrome and Linux dominate training.
- **Function-level only:** no cross-function or cross-file bugs.
- **No hyperparameter tuning; no macro expansion.**

**Bonus: Why did the extra edges not help?**
Over five seeds AST-only and all-edges are tied, so the extra edges neither help nor
hurt measurably. Two likely reasons:
- Our GraphSAGE layers **ignore edge types**, so next-statement and variable-use
  edges arrive as untyped extra neighbours.
- Name matching links unrelated variables that share a name (`i`, `len`, `buf`).

Testing typed edges would need an edge-type-aware layer such as R-GCN.

**Bonus: You tried predicting the CWE type. Why didn't it work, and is that a failure?**
The type head reached macro-F1 0.151 against 0.117 for always guessing "Other", but
its accuracy (0.495) was below that constant guess (0.544). A TF-IDF text model did
no better (0.118). So the bottleneck is the labels, not the GNN:
- Only ~900 vulnerable training functions have a CWE.
- The labels drift across projects and years: val/test are dominated by CWE-125
  and CWE-190, which are rare in train, and older CVEs used the broad CWE-119.

We also found that joint training stopped too early for the second head (the risk
task peaks after one epoch), so we added a second stage that trains only the head
on frozen embeddings. Reporting a negative result honestly, and *not* showing an
unreliable label to users, is the right engineering call.

---

## Engineering and ML challenges solved (good for a "challenges" slide)

1. **Unbalanced project split.** Assigning 80% of *projects* to train gave a test
   set of only 819 functions, because a few huge projects hold most of the code.
   Fixed with a greedy packer that keeps projects whole but balances *function
   counts* to 80/10/10.
2. **No flaw-line column in the dataset mirror.** Ground-truth vulnerable lines were
   recovered by diffing `func_before` against `func_after`.
3. **Probabilities stuck near 50%.** The weighted loss biases scores upwards and
   temperature scaling cannot shift them. Fixed with Platt scaling (ECE 0.446 →
   0.014) without retraining and without changing the flagged functions.
4. **An evaluation bug with tied scores.** Precision@Top-10% broke ties by row
   order, which made Flawfinder look worse than random (0.023). Fixed with
   tie-aware counting (0.116).
5. **A missing baseline.** Adding a lines-of-code baseline revealed that length
   explains most of the model's performance. This changed the project's conclusion
   and is reported openly.
6. **GitHub's Content-Security-Policy** blocks page scripts from calling a local
   server, so all network calls go through the extension's background service
   worker.
7. **`localhost` vs `127.0.0.1`.** Chrome resolved `localhost` to IPv6 `::1` while
   uvicorn listened on IPv4 only, so requests hung for 30+ seconds. Fixed by using
   `127.0.0.1` with a 10-second timeout and precise error messages.
8. **C macros confused the function-name extractor** (`static NORETURN void die(...)`
   was reported as `void`). Fixed by reading the name from the function declarator.
9. **GitHub's React code view** keeps only part of a long file in the page and
   re-renders lines, wiping injected marks. Fixed by re-applying marks on scroll and
   on a 1.5-second timer, with idempotent marking.
10. **Stale cache after code changes**, fixed by adding a scanner version to the
    SQLite cache key.
