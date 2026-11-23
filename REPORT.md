# Report material: abstract, viva questions, challenges

## Abstract

Rule-based static analyzers for C/C++ miss structural bugs and raise many false
alarms, while text-based machine-learning detectors discard the tree structure and
execution order of code. This project presents a vulnerability scanner that
represents each C/C++ function as a graph, built from its tree-sitter Abstract
Syntax Tree plus next-statement and name-matched variable-use edges, and
classifies it with a Hybrid GraphSAGE network trained from scratch. The hybrid
model concatenates an attention-plus-mean-pooled graph embedding with six
hand-crafted code metrics (size, cyclomatic complexity, pointer operations,
risky-API calls, nesting depth, parameters). On a project-disjoint split of the
Big-Vul dataset (30,000 functions, about 6% vulnerable), it reaches a PR-AUC of
0.208 and a Precision@Top-10% of 0.203, roughly double the best of three baselines
(Flawfinder, TF-IDF with Logistic Regression, and a risky-API rule; best PR-AUC
0.108). Ablations show that the hand-crafted metrics are the largest single
contributor and that SAGEConv slightly outperforms GCNConv. For explanations, node
importance is computed as attention weight times gradient saliency and aggregated
over single-line nodes inside the function body. The single top-ranked line is a
line changed by the real fix for 21.2% of vulnerable test functions (8.6% for a
random line), and one of the top three for 38.0% (19.2% random). The model is served by a FastAPI backend
with SQLite caching (under 0.4 s per file on CPU) and surfaced on GitHub through a
Manifest V3 Chrome extension that highlights risky functions and lines and collects
false-positive feedback.

---

## Likely viva questions (with short answers)

**1. Why GraphSAGE, and not a plain GCN?**
GraphSAGE updates a node by **aggregating its neighbours** (we use the mean) and
combining that with the node's own vector, using **separate learned weights** for
"self" and "neighbours". So a node keeps its own identity (e.g. "this is a `strcpy`
call") instead of being blurred into its surroundings. It is also **inductive**: it
learns a function that works on graphs never seen in training, which is exactly our
case, since every scanned function is a new graph. GCN averages a node together with
its neighbours using degree-based normalisation and one shared weight matrix. In
our ablation, SAGEConv scored PR-AUC 0.208 vs 0.194 for GCNConv: a small edge from a
single run. (GraphSAGE was originally proposed with neighbour *sampling* for huge
graphs; our function graphs are small, so we aggregate all neighbours.)

**2. Why split the data by project instead of randomly?**
Functions from the same project share coding style, macros, helper functions and
naming. With a random split, near-identical functions land in both train and test,
and the model can score well by memorising a project's style. That is
**information leakage**. Splitting by project means the test set contains only
projects the model never saw, which matches the real use case: scanning a new
GitHub repository. It makes the numbers lower but honest.

**3. Why PR-AUC instead of accuracy?**
Only about 6% of functions are vulnerable. A model that always says "safe" gets
about 94% accuracy while finding zero bugs. PR-AUC (area under the
precision-recall curve) only looks at how well the **vulnerable class** is ranked,
across all thresholds, so the big safe majority cannot inflate it. A random model
scores about the positive rate (0.064 here). We also report Precision@Top-10%,
which answers the practical question: "if a reviewer checks the 10% riskiest
functions, how many are real bugs?"

**4. What do the hybrid (hand-crafted) features add?**
They give the classifier cheap, global facts that a 3-layer GNN sees only locally:
total size, complexity, pointer use, and risky calls. Vulnerable functions score
higher on every metric: about 2-3.3× for size, complexity, pointer operations and
risky calls, and about 1.3× for nesting depth and parameters. Removing them dropped PR-AUC from 0.208 to 0.133, the
largest effect in any ablation. The caveat is that several metrics track function
size, so part of the gain is "big functions are riskier". The GNN still adds
structure on top: the pure-GNN ablation (0.133) beats every baseline.

**5. How are the line-level explanations computed, and can you trust them?**
Each node gets **attention weight** (how much the pooling layer relied on it) ×
**gradient saliency** (how much the risk score changes if that node's features
change, from one backward pass). Node scores are summed per source line and the top
3 lines are returned, with a template reason when a line matches a known pattern
(strcpy/strcat, memcpy, use after free, malloc without NULL check). They show
**where the model looked, not proof of a bug**. Two rules made them much more
useful:
- Only **single-line nodes** vote. Big multi-line nodes, like the whole function,
  would otherwise dump their importance on the signature line.
- Only **body lines** are ranked.

A third, minor rule lists lines matching a known dangerous pattern first within
the top 3. All rules were chosen on the validation set, not the test set. On test,
the single named line is a real fix line 21.2% of the time (8.6% random, about 2.5×),
and the top 3 contain one 38.0% of the time (19.2% random). Useful, but far from
perfect.

**6. What are the main limitations?**
- Variable-use edges match identifiers **by name**, not real data flow, which is
  probably why AST-only edges did slightly better (0.228 vs 0.208).
- **Function-level only**: no cross-function or cross-file bugs.
- **Labels are noisy** because they come from vulnerability-fixing commits.
- **Low confidence**: calibrated scores cluster near 0.5, so almost everything is
  "Medium".
- **No macro expansion**, so macro-heavy code parses imperfectly.
- **Results come from one training run on a 30k subset**, so small differences are
  within noise.

**Bonus: Why did the extra edges not help?**
Name matching links unrelated variables that share a name (`i`, `len`, `buf`) and
adds noisy shortcuts between distant parts of the graph. The AST already encodes a
lot of local order. Proper scope-aware def-use edges are listed as future work.

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

## Engineering challenges solved (good for a "challenges" slide)

1. **Unbalanced project split.** Assigning 80% of *projects* to train gave a test
   set of only 819 functions, because a few huge projects hold most of the code.
   Fixed with a greedy packer that keeps projects whole but balances *function
   counts* to 80/10/10.
2. **No flaw-line column in the dataset mirror.** Ground-truth vulnerable lines were
   recovered by diffing `func_before` against `func_after`.
3. **GitHub's Content-Security-Policy** blocks page scripts from calling a local
   server, so all network calls go through the extension's background service
   worker.
4. **`localhost` vs `127.0.0.1`.** Chrome resolved `localhost` to IPv6 `::1` while
   uvicorn listened on IPv4 only, so requests hung for 30+ seconds. Fixed by using
   `127.0.0.1` with a 10-second timeout and precise error messages.
5. **C macros confused the function-name extractor** (`static NORETURN void die(...)`
   was reported as `void`). Fixed by reading the name from the function declarator.
6. **GitHub's React code view** keeps only part of a long file in the page and
   re-renders lines, wiping injected marks. Fixed by re-applying marks on scroll and
   on a 1.5-second timer, with idempotent marking.
7. **Stale cache after code changes**, fixed by adding a scanner version to the
   SQLite cache key.
