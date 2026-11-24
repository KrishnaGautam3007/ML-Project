"""
Probability calibration for the risk model.

Why this exists
---------------
The model is trained with a weighted loss (pos_weight = #safe / #vulnerable,
about 18.6) so that the rare vulnerable class is not ignored. A side effect is
that the raw logits are shifted upwards by roughly ln(18.6) ~ 2.9: the model's
"probabilities" are far too high for a class that is only ~6% of the data.

Temperature scaling (sigmoid(z / T)) can only STRETCH or SQUASH the logits, not
SHIFT them, so the best it could do here was squash every score towards 0.5
(all test probabilities ended up between 0.40 and 0.64).

Platt scaling fits sigmoid(a * z + b) on the validation set: `a` rescales and
`b` shifts, which removes the bias introduced by the weighted loss. Both are
monotonic, so the ranking of functions (and therefore PR-AUC and
Precision@Top-10%) does not change - only the probability values do.

A calibration is stored as a small dict so it can live inside the checkpoint:
    {"type": "platt", "a": ..., "b": ...}
    {"type": "temperature", "T": ...}        (older checkpoints)
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt(logits, y) -> dict:
    """Fit sigmoid(a*z + b) to validation logits z and labels y."""
    z = np.asarray(logits, dtype=float).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000)   # ~no regularisation: plain Platt fit
    lr.fit(z, np.asarray(y).astype(int))
    return {"type": "platt", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def apply(calibration: dict, logits):
    """Turn raw logits into calibrated probabilities (works on scalars and arrays)."""
    z = np.asarray(logits, dtype=float)
    if calibration["type"] == "platt":
        s = calibration["a"] * z + calibration["b"]
    elif calibration["type"] == "temperature":
        s = z / calibration["T"]
    else:
        raise ValueError(f"unknown calibration type {calibration['type']!r}")
    p = 1.0 / (1.0 + np.exp(-s))
    return float(p) if p.ndim == 0 else p


def to_logit_threshold(calibration: dict, prob_threshold: float) -> float:
    """The logit z* where the calibrated probability equals prob_threshold."""
    s = math.log(prob_threshold / (1 - prob_threshold))
    if calibration["type"] == "platt":
        return (s - calibration["b"]) / calibration["a"]
    return s * calibration["T"]


def from_checkpoint(ckpt: dict) -> dict:
    """Read the calibration from a saved checkpoint (old ones only had T)."""
    if "calibration" in ckpt:
        return ckpt["calibration"]
    return {"type": "temperature", "T": float(ckpt["temperature"])}


def expected_calibration_error(probs, y, n_bins: int = 10) -> float:
    """ECE with equal-count bins: average |mean predicted - observed rate|."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(probs)
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ece += len(chunk) / len(probs) * abs(probs[chunk].mean() - y[chunk].mean())
    return float(ece)
