"""
Flowchart diagrams for the project report (drawn as SVG, saved as PNG).

Two diagrams that the explainer PDF does not have:
  - scan_flow:        what happens when a C/C++ file is opened on GitHub
  - explanation_flow: how the "most likely vulnerable line" is chosen

Each diagram is plain SVG built from a few helper functions, then rendered to
a sharp PNG with headless Chrome (3x pixel density).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

# colours (same palette as the explainer PDF)
GREY_FILL, GREY_LINE = "#f4f3ef", "#c3c2b7"
BLUE, BLUE_SOFT = "#2a78d6", "#e8f1fc"
AMBER_SOFT, AMBER = "#fdf4e3", "#eda100"
RED, RED_SOFT = "#d03b3b", "#fbeaea"
INK, ARROW = "#0b0b0b", "#898781"
FONT = "Segoe UI, system-ui, sans-serif"

LINE_H = 16  # text line height in px


# -----------------------------------------------------------------------------
# small SVG building blocks
# -----------------------------------------------------------------------------

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(cx, cy, lines, color=INK, bold=False, size=13):
    """Centred multi-line text around (cx, cy)."""
    top = cy - (len(lines) - 1) * LINE_H / 2
    out = []
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{cx}" y="{top + i * LINE_H + 4.5}" text-anchor="middle" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">'
            f"{_esc(line)}</text>")
    return "".join(out)


def box(x, y, w, h, lines, fill=GREY_FILL, stroke=GREY_LINE, color=INK, bold=False, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"/>'
            + _text(x + w / 2, y + h / 2, lines, color, bold))


def pill(cx, y, w, h, lines, fill=BLUE, color="#ffffff"):
    return box(cx - w / 2, y, w, h, lines, fill=fill, stroke=fill, color=color,
               bold=True, rx=h / 2)


def diamond(cx, cy, hw, hh, lines):
    pts = f"{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}"
    return (f'<polygon points="{pts}" fill="{AMBER_SOFT}" stroke="{AMBER}" stroke-width="1.5"/>'
            + _text(cx, cy, lines, bold=True))


def arrow(points, label=None, label_at=None):
    """Polyline arrow through the given (x, y) points; optional label."""
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    out = (f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="1.8" '
           f'marker-end="url(#head)"/>')
    if label:
        lx, ly = label_at
        out += (f'<text x="{lx}" y="{ly}" font-size="12" font-weight="700" '
                f'fill="{INK}" text-anchor="middle">{_esc(label)}</text>')
    return out


def svg(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{FONT}">'
            '<defs><marker id="head" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
            f'orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{ARROW}"/></marker></defs>'
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>{body}</svg>')


# -----------------------------------------------------------------------------
# Diagram 1: scan request flow
# -----------------------------------------------------------------------------

def scan_flow():
    W, H = 620, 706
    cx = 190                      # main column centre
    bx, bw = 40, 300              # main column box x / width
    rx, rw = 370, 220             # right column (cache hit)
    rcx = rx + rw / 2
    parts = [
        pill(cx, 10, 300, 36, ["User opens a C/C++ file on GitHub"]),
        arrow([(cx, 46), (cx, 68)]),
        box(bx, 70, bw, 44, ["Extension (content.js) detects the", "file and builds its raw URL"]),
        arrow([(cx, 114), (cx, 136)]),
        box(bx, 138, bw, 44, ["Background worker downloads", "the raw code from GitHub"]),
        arrow([(cx, 182), (cx, 204)]),
        box(bx, 206, bw, 44, ["POST /scan to the FastAPI server", "(127.0.0.1:8000)"]),
        arrow([(cx, 250), (cx, 262)]),
        diamond(cx, 304, 120, 42, ["Result already in", "the SQLite cache?"]),
        # yes -> right
        arrow([(cx + 120, 304), (rx - 2, 304)], "Yes", (cx + 150, 296)),
        box(rx, 282, rw, 44, ["Return the cached result", "(about 2 ms)"]),
        # no -> down
        arrow([(cx, 346), (cx, 372)], "No", (cx + 18, 364)),
        box(bx, 374, bw, 44, ["Split the file into functions", "(tree-sitter)"]),
        arrow([(cx, 418), (cx, 440)]),
        box(bx, 442, bw, 44, ["Each function: graph → features →", "Hybrid GraphSAGE → risk score"],
            fill=BLUE_SOFT, stroke=BLUE),
        arrow([(cx, 486), (cx, 508)]),
        box(bx, 510, bw, 44, ["Rank lines (attention × saliency)", "and attach a reason"],
            fill=BLUE_SOFT, stroke=BLUE),
        arrow([(cx, 554), (cx, 576)]),
        box(bx, 578, bw, 44, ["Save the result in the cache", "and return JSON"]),
        arrow([(cx, 622), (cx, 646)]),
        arrow([(rcx, 326), (rcx, 646)]),
        pill((bx + rx + rw) / 2, 648, rx + rw - bx, 44,
             ["Extension draws risk badges, “Likely vulnerable line” labels and the side panel"],
             fill=RED),
    ]
    return svg(W, H, "".join(parts)), (W, H)


# -----------------------------------------------------------------------------
# Diagram 2: how the most likely vulnerable line is chosen
# -----------------------------------------------------------------------------

def explanation_flow():
    W, H = 680, 590
    cx = 340
    parts = [
        pill(cx, 10, 220, 36, ["One C/C++ function"]),
        arrow([(cx, 46), (cx, 66)]),
        box(cx - 200, 68, 400, 44, ["Build its graph: 166 numbers per node", "+ 6 function metrics"]),
        arrow([(cx - 60, 112), (180, 144)]),
        arrow([(cx + 60, 112), (500, 144)]),
        box(40, 146, 280, 58, ["Forward pass:", "attention weight α for every node"],
            fill=BLUE_SOFT, stroke=BLUE),
        box(360, 146, 280, 58, ["Backward pass: gradient saliency", "|∂ risk / ∂ node| for every node"],
            fill=BLUE_SOFT, stroke=BLUE),
        arrow([(180, 204), (cx - 70, 238)]),
        arrow([(500, 204), (cx + 70, 238)]),
        box(cx - 160, 240, 320, 40, ["Node importance = α × saliency"],
            fill=AMBER_SOFT, stroke=AMBER, bold=True),
        arrow([(cx, 280), (cx, 302)]),
        box(cx - 230, 304, 460, 50, ["Add up importance per source line, counting only",
                                     "single-line nodes and only lines inside the body"]),
        arrow([(cx, 354), (cx, 376)]),
        box(cx - 160, 378, 320, 40, ["Keep the 3 highest-scoring lines"]),
        arrow([(cx, 418), (cx, 440)]),
        box(cx - 230, 442, 460, 50, ["Lines with a known danger (strcpy, memcpy, malloc…)",
                                     "go first; each line gets a short reason"]),
        arrow([(cx, 492), (cx, 516)]),
        pill(cx, 518, 440, 44, ["#1 = most likely vulnerable line, #2–#3 = suspicious"],
             fill=RED),
    ]
    return svg(W, H, "".join(parts)), (W, H)


# -----------------------------------------------------------------------------
# rendering
# -----------------------------------------------------------------------------

def render(svg_text: str, size, out_png: Path, browser: str):
    """Screenshot the SVG with headless Chrome at 3x pixel density."""
    w, h = size
    tmp = Path(tempfile.mkdtemp(prefix="diagram_"))
    try:
        page = tmp / "d.html"
        page.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
            "html,body{margin:0;padding:0;background:#fff;overflow:hidden}"
            "svg{display:block}</style></head><body>" + svg_text + "</body></html>",
            encoding="utf-8")
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run",
               f"--user-data-dir={tmp / 'profile'}", "--hide-scrollbars",
               f"--window-size={w},{h}", "--force-device-scale-factor=3",
               # Chrome needs an ABSOLUTE output path, or the file lands elsewhere
               f"--screenshot={Path(out_png).resolve()}", page.as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_all(out_dir: Path, browser: str):
    for name, builder in [("report_fig_scan_flow.png", scan_flow),
                          ("report_fig_explain_flow.png", explanation_flow)]:
        svg_text, size = builder()
        render(svg_text, size, out_dir / name, browser)
