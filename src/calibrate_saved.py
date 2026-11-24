"""
Re-calibrate the ALREADY TRAINED main model with Platt scaling (no retraining).

The saved model (models/hybrid_gnn.pt) was calibrated with temperature scaling,
which squashed every probability to 0.40-0.64 (see calibration.py for why).
This script:
  1. computes the model's raw logits on the validation set,
  2. fits Platt scaling sigmoid(a*z + b) on them,
  3. converts the old decision threshold so the SAME functions are still
     flagged (identical precision / recall / F1 on test), and
  4. saves the new calibration into the checkpoint (the old one is kept too).

The model weights, the ranking, PR-AUC and Precision@Top-10% do not change.

Run:  python src/calibrate_saved.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

import calibration
from explain import load_model, MODELS_DIR
from features import load_type_vocab
from gnn_data import load_graphs, prepare_graphs, make_loader

CKPT = os.path.join(MODELS_DIR, "hybrid_gnn.pt")


@torch.no_grad()
def logits_for(model, split, vocab):
    graphs = prepare_graphs(load_graphs(split), vocab)
    zs, ys = [], []
    for batch in make_loader(graphs, 128):
        zs.append(model(batch))
        ys.append(batch.y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    old_calib = calibration.from_checkpoint(ckpt)
    if old_calib["type"] == "platt":
        print("Checkpoint is already Platt-calibrated; nothing to do.")
        return
    old_thr = float(ckpt["threshold"])

    model, vocab, _, _, _ = load_model()
    z_val, y_val = logits_for(model, "val", vocab)
    new_calib = calibration.fit_platt(z_val, y_val)

    # keep the same decision boundary in logit space -> same flagged functions
    z_star = calibration.to_logit_threshold(old_calib, old_thr)
    new_thr = float(calibration.apply(new_calib, z_star))

    for name, c, t in (("old (temperature)", old_calib, old_thr),
                       ("new (Platt)", new_calib, new_thr)):
        p = calibration.apply(c, z_val)
        print(f"{name:18}: val probs {p.min():.3f}-{p.max():.3f}, mean {p.mean():.3f} "
              f"(true rate {y_val.mean():.3f}), ECE {calibration.expected_calibration_error(p, y_val):.3f}, "
              f"threshold {t:.3f}")

    ckpt["previous_calibration"] = {**old_calib, "threshold": old_thr}
    ckpt["calibration"] = new_calib
    ckpt["threshold"] = new_thr
    ckpt.pop("temperature", None)
    torch.save(ckpt, CKPT)
    print(f"Saved Platt calibration a={new_calib['a']:.4f}, b={new_calib['b']:.4f}, "
          f"threshold {new_thr:.4f} -> {CKPT}")


if __name__ == "__main__":
    main()
