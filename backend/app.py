"""
Phase 7 - FastAPI backend.

Endpoints:
  POST /scan      {filename, code}  -> list of functions with risk + top lines
  POST /feedback  {file_hash, function_name, line, is_false_positive}
  GET  /health    quick "is it up?" check

Design notes:
  - The model is loaded ONCE at startup (loading it per request would be slow).
  - Scanning runs on CPU.
  - Results are cached in SQLite keyed by the SHA-256 of the file content, so the
    exact same file is never scanned twice.
  - CORS is enabled for github.com so the Chrome extension (Phase 8) can call us.

Run:
    uvicorn backend.app:app --port 8000      (from the project root)
"""

from __future__ import annotations

import os
import sys
import json
import time
import sqlite3
import hashlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# make the src/ modules importable
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from parsing import iter_functions
from explain import load_model, explain_function

DB_PATH = os.path.join(HERE, "cache.db")

# Bump this whenever the model or the parsing/explanation code changes, so old
# cached results (made by older code) are not served for the same file.
SCANNER_VERSION = "6"    # 6: Platt-calibrated probabilities


# ---------------------------------------------------------------------------
# request/response shapes
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    filename: str
    code: str


class FeedbackRequest(BaseModel):
    file_hash: str
    function_name: str
    line: int
    is_false_positive: bool


# ---------------------------------------------------------------------------
# tiny SQLite helpers
# ---------------------------------------------------------------------------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_cache (
                file_hash   TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at  REAL NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash     TEXT NOT NULL,
                function_name TEXT,
                line          INTEGER,
                is_false_positive INTEGER,
                created_at    REAL NOT NULL
            )""")


def cache_get(file_hash: str):
    with db() as conn:
        row = conn.execute("SELECT result_json FROM scan_cache WHERE file_hash=?",
                           (file_hash,)).fetchone()
    return json.loads(row["result_json"]) if row else None


def cache_put(file_hash: str, result: dict):
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scan_cache (file_hash, result_json, created_at) VALUES (?,?,?)",
            (file_hash, json.dumps(result), time.time()))


# ---------------------------------------------------------------------------
# app + model
# ---------------------------------------------------------------------------

app = FastAPI(title="GNN Vulnerability Scanner")

# CORS: allow GitHub pages (and localhost for local testing) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://(.*\.)?github\.com|http://localhost(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

# loaded in startup
MODEL = VOCAB = STATS = None
CALIBRATION = THRESHOLD = None


@app.on_event("startup")
def _startup():
    global MODEL, VOCAB, STATS, CALIBRATION, THRESHOLD
    init_db()
    MODEL, VOCAB, STATS, CALIBRATION, THRESHOLD = load_model()
    print("Model loaded. Ready to scan on CPU.")


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/scan")
def scan(req: ScanRequest):
    t0 = time.time()
    # file_hash identifies the FILE (sent back to the extension, used by /feedback);
    # cache_key also includes the scanner version so code changes bust the cache.
    file_hash = hashlib.sha256(req.code.encode("utf-8", errors="ignore")).hexdigest()
    cache_key = f"v{SCANNER_VERSION}:{file_hash}"

    # 1. cache hit?
    cached = cache_get(cache_key)
    if cached is not None:
        cached["cached"] = True
        cached["scan_time_ms"] = round((time.time() - t0) * 1000, 1)
        return cached

    # 2. split the file into functions and score each one
    functions = []
    for fn in iter_functions(req.code):
        exp = explain_function(fn.code, MODEL, VOCAB, STATS, CALIBRATION, THRESHOLD)
        # explain returns line numbers relative to the function; shift them to
        # be absolute within the whole file.
        # (top_lines are already ranked: index 0 = most likely vulnerable line)
        top_lines = [
            {"line": fn.start_line + (t["line"] - 1), "reason": t["reason"],
             "text": t["text"][:200]}
            for t in exp["top_lines"]
        ]
        functions.append({
            "name": fn.name,
            "start_line": fn.start_line,
            "end_line": fn.end_line,
            "risk_score": exp["risk_score"],
            "band": exp["band"],
            "top_lines": top_lines,
        })

    result = {
        "filename": req.filename,
        "file_hash": file_hash,
        "functions": functions,
        "cached": False,
    }

    # 3. cache + log time
    cache_put(cache_key, result)
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    result["scan_time_ms"] = elapsed_ms
    print(f"[scan] {req.filename}: {len(functions)} functions in {elapsed_ms} ms")
    return result


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    with db() as conn:
        conn.execute(
            "INSERT INTO feedback (file_hash, function_name, line, is_false_positive, created_at)"
            " VALUES (?,?,?,?,?)",
            (req.file_hash, req.function_name, req.line, int(req.is_false_positive), time.time()))
    return {"status": "recorded"}
