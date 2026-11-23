"""
Phase 3 (part 2) - Turn one function into a graph (PyTorch Geometric Data).

We keep only NAMED tree-sitter nodes. Each node remembers:
  - its type       (e.g. "if_statement", "identifier")
  - its token text (only for leaf nodes, e.g. the actual variable name)
  - its start line (1-based; we need this for line-level explanations in Phase 6)

We add THREE kinds of edges. All are stored undirected (we add both u->v and
v->u) because GraphSAGE passes messages both ways. Each edge also carries an
integer "edge type" so later code could tell them apart:

  0 = AST edge          : parent  <-> child in the syntax tree
  1 = next-statement    : statement  <->  the next statement in the same block
                          (a cheap stand-in for "runs after")
  2 = variable-use      : one use of a name  <->  the next use of the SAME name
                          (a cheap stand-in for data flow; matched by name only)

At this stage we do NOT compute numeric node features yet - that is Phase 4.
We only build the structure and keep the raw per-node info on the Data object.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from parsing import parse

# edge type ids
EDGE_AST = 0
EDGE_NEXT = 1
EDGE_VARUSE = 2
EDGE_TYPE_NAMES = {EDGE_AST: "AST", EDGE_NEXT: "NEXT", EDGE_VARUSE: "VARUSE"}

# tree-sitter node types that count as "statements" for next-statement edges
STATEMENT_TYPES = {
    "expression_statement", "declaration", "if_statement", "for_statement",
    "while_statement", "do_statement", "switch_statement", "return_statement",
    "break_statement", "continue_statement", "compound_statement",
    "labeled_statement", "goto_statement", "case_statement",
}


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def build_graph(code: str, label: int | None = None, lang: str | None = None) -> Data:
    """Parse one function and return a PyG Data graph (no numeric features yet)."""
    tree, lang_used = parse(code, lang)
    source_bytes = code.encode("utf-8", errors="ignore")
    root = tree.root_node

    # ---- 1. collect named nodes and give each an index (pre-order walk) ----
    node_list = []          # index -> tree-sitter node
    index_of = {}           # tree-sitter node.id -> index

    stack = [root]
    while stack:
        node = stack.pop()
        if node.id in index_of:
            continue
        index_of[node.id] = len(node_list)
        node_list.append(node)
        # push named children only (we ignore pure punctuation)
        for child in reversed(node.named_children):
            stack.append(child)

    n = len(node_list)

    node_types = [nd.type for nd in node_list]
    # token text only for leaves (named nodes with no named children)
    node_tokens = [
        _node_text(nd, source_bytes) if nd.named_child_count == 0 else ""
        for nd in node_list
    ]
    node_lines = [nd.start_point[0] + 1 for nd in node_list]
    # end line too: lets explanations ignore big multi-line "container" nodes
    # (the whole function, a { } block) that would otherwise pile their
    # importance onto their first line - usually the function signature.
    node_end_lines = [nd.end_point[0] + 1 for nd in node_list]

    edges = []       # list of (u, v)
    edge_types = []  # parallel list of edge-type ids

    def add_edge(u: int, v: int, etype: int):
        # store undirected: both directions, same type
        edges.append((u, v)); edge_types.append(etype)
        edges.append((v, u)); edge_types.append(etype)

    # ---- 2. AST edges: parent <-> named child ----
    for parent in node_list:
        p = index_of[parent.id]
        for child in parent.named_children:
            add_edge(p, index_of[child.id], EDGE_AST)

    # ---- 3. next-statement edges: consecutive statements in a block ----
    for parent in node_list:
        kids = [c for c in parent.named_children if c.type in STATEMENT_TYPES]
        for a, b in zip(kids, kids[1:]):
            add_edge(index_of[a.id], index_of[b.id], EDGE_NEXT)

    # ---- 4. variable-use edges: chain uses of the same identifier name ----
    # group identifier node indices by their text, in source order
    name_to_indices: dict[str, list[int]] = {}
    for idx, nd in enumerate(node_list):
        if nd.type == "identifier":
            name = node_tokens[idx]
            if name:
                name_to_indices.setdefault(name, []).append(idx)
    for name, idxs in name_to_indices.items():
        # idxs are already in pre-order; link each occurrence to the next one.
        # The FIRST occurrence approximates the declaration (name-based only).
        for a, b in zip(idxs, idxs[1:]):
            add_edge(a, b, EDGE_VARUSE)

    # ---- 5. pack into a PyG Data object ----
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    data = Data(edge_index=edge_index)
    data.edge_type = edge_type
    data.num_nodes = n
    data.line = torch.tensor(node_lines, dtype=torch.long)
    data.end_line = torch.tensor(node_end_lines, dtype=torch.long)
    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)
    # keep raw info for Phase 4 (features) and Phase 6 (explanations).
    # stored as plain python lists; fine because we save graphs one by one.
    data.node_types = node_types
    data.node_tokens = node_tokens
    data.lang = lang_used
    return data


# ---------------------------------------------------------------------------
# A tiny human-readable view of a graph, so we can check it by eye
# ---------------------------------------------------------------------------

def print_example_graph(data: Data, max_nodes: int = 40):
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} directed "
          f"edges (= {data.edge_index.size(1)//2} undirected), lang={data.lang}")
    print("\nNodes (index: type  @line  'token'):")
    for i in range(min(data.num_nodes, max_nodes)):
        tok = data.node_tokens[i]
        tok = f"  '{tok}'" if tok else ""
        print(f"  {i:>3}: {data.node_types[i]:<22} @L{int(data.line[i])}{tok}")
    if data.num_nodes > max_nodes:
        print(f"  ... ({data.num_nodes - max_nodes} more nodes)")

    print("\nEdges by type:")
    ei = data.edge_index
    et = data.edge_type
    # only print one direction (u < v) to keep it short
    seen = set()
    for k in range(ei.size(1)):
        u, v = int(ei[0, k]), int(ei[1, k])
        key = (min(u, v), max(u, v), int(et[k]))
        if key in seen:
            continue
        seen.add(key)
        print(f"  {u:>3} <-> {v:<3} [{EDGE_TYPE_NAMES[int(et[k])]}]")
        if len(seen) >= 60:
            print("  ... (more edges)")
            break


if __name__ == "__main__":
    sample = """int copy(char *src) {
    char buf[10];
    strcpy(buf, src);
    return 0;
}"""
    g = build_graph(sample, label=1)
    print_example_graph(g)
