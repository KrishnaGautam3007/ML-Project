"""
Phase 3 (part 1) - Parsing C/C++ with tree-sitter.

Two jobs:
  1. Give us a ready-to-use tree-sitter Parser for C or C++.
  2. Split a whole source FILE into its individual functions (used by the
     backend in Phase 7/8 when we scan a real file). For the Big-Vul dataset we
     already have one function per row, so there we just parse the function text
     directly.

tree-sitter turns source code into a "concrete syntax tree": every piece of the
code (a function, an if, a variable name, a `;`) becomes a node. "Named" nodes
are the meaningful ones (identifier, if_statement, ...) as opposed to anonymous
punctuation tokens. We keep only named nodes later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tree_sitter import Language, Parser
import tree_sitter_c as ts_c
import tree_sitter_cpp as ts_cpp

# Build the two Language objects once (cheap to reuse).
C_LANGUAGE = Language(ts_c.language())
CPP_LANGUAGE = Language(ts_cpp.language())


def get_parser(lang: str = "c") -> Parser:
    """Return a tree-sitter Parser for 'c' or 'cpp'."""
    language = CPP_LANGUAGE if lang == "cpp" else C_LANGUAGE
    return Parser(language)


def guess_lang(code: str) -> str:
    """Very small heuristic to choose the C++ grammar when the code looks C++.

    Big-Vul is mostly C, so we default to C and only switch to C++ when we see
    obvious C++ markers. This is a heuristic, not a guarantee.
    """
    markers = ("::", "template<", "template <", "std::", "class ", "namespace ",
               "public:", "private:", "nullptr", "new ", "delete ")
    return "cpp" if any(m in code for m in markers) else "c"


def parse(code: str, lang: str | None = None):
    """Parse a code string and return (tree, lang_used)."""
    if lang is None:
        lang = guess_lang(code)
    parser = get_parser(lang)
    tree = parser.parse(code.encode("utf-8", errors="ignore"))
    return tree, lang


# ---------------------------------------------------------------------------
# Splitting a whole file into functions (for the backend, Phase 7/8)
# ---------------------------------------------------------------------------

@dataclass
class FunctionSpan:
    name: str
    start_line: int      # 1-based, inclusive
    end_line: int        # 1-based, inclusive
    code: str            # the function's source text


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


NAME_TYPES = ("identifier", "field_identifier", "qualified_identifier",
              "destructor_name", "operator_name")


def _find_function_name(def_node, source_bytes: bytes) -> str:
    """Dig out the function's name from a function_definition node.

    The name is the thing attached directly to the parameter list, i.e. the
    'declarator' of the `function_declarator` node. We must NOT just take the
    first identifier we see: with macros like `static NORETURN void die(...)`
    tree-sitter cannot know NORETURN is a macro, and the first identifier would
    be 'NORETURN' or 'void' instead of 'die'.
    """
    # 1) breadth-first search for the function_declarator (skipping the body)
    queue = [def_node]
    while queue:
        n = queue.pop(0)
        if n.type == "function_declarator":
            d = n.child_by_field_name("declarator")
            # unwrap wrappers such as parenthesized/pointer declarators
            while d is not None and d.type not in NAME_TYPES:
                d = d.child_by_field_name("declarator") or (
                    d.named_children[0] if d.named_children else None)
            if d is not None:
                return _node_text(d, source_bytes)
        if n.type == "compound_statement":
            continue                      # never look inside the body
        queue.extend(n.children)

    # 2) fallback: the identifier right before the first '(' in the header
    header = _node_text(def_node, source_bytes).split("{", 1)[0]
    m = re.search(r"([A-Za-z_]\w*)\s*\(", header)
    return m.group(1) if m else "<anonymous>"


def iter_functions(code: str, lang: str | None = None):
    """Yield FunctionSpan for every top-level function definition in `code`."""
    tree, lang_used = parse(code, lang)
    source_bytes = code.encode("utf-8", errors="ignore")
    root = tree.root_node

    # Walk the tree looking for 'function_definition' nodes.
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            name = _find_function_name(node, source_bytes)
            yield FunctionSpan(
                name=name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                code=_node_text(node, source_bytes),
            )
            # don't descend into nested function bodies (rare in C)
            continue
        stack.extend(node.children)


if __name__ == "__main__":
    # tiny self-test
    sample = """
    int add(int a, int b) {
        int c = a + b;
        return c;
    }

    void greet() {
        printf("hi");
    }
    """
    for fn in iter_functions(sample):
        print(f"{fn.name:10} lines {fn.start_line}-{fn.end_line}")
