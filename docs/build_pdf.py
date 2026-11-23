"""
Build docs/GNN_Vulnerability_Scanner_Explained.pdf from docs/project_explained.html.

It prints the HTML page to PDF with headless Chrome (or Edge), which keeps the
colours, diagrams and charts exactly as designed.

Run from the project root:
    python docs/build_pdf.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "project_explained.html"
OUTPUT = HERE / "GNN_Vulnerability_Scanner_Explained.pdf"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str:
    for path in BROWSERS:
        if os.path.exists(path):
            return path
    raise SystemExit("Could not find Chrome or Edge. Install one, or edit BROWSERS.")


def main():
    browser = find_browser()
    # a throw-away profile, so we never touch (or clash with) your normal browser
    profile = tempfile.mkdtemp(prefix="pdf_profile_")
    try:
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",          # no date/URL printed on each page
            "--virtual-time-budget=10000",     # let the chart script + image load
            f"--print-to-pdf={OUTPUT}",
            SOURCE.as_uri(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    finally:
        shutil.rmtree(profile, ignore_errors=True)

    print(f"PDF written: {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
