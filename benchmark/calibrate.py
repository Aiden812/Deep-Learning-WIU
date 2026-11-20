"""Step 0 of the v1.3 plan: re-pick each model's yes/no threshold on photos it never saw, with no retraining.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/calibrate.py

Why: on MetaDent DS1 the detectors rank photos well (AUROC 0.83-0.88) but their saved cut (conf 0.25) only catches
27-37% of problem photos. A threshold picked on our own validation data doesn't carry over to other cameras.

How: for every model that gives a photo-level problem score, take the MetaDent **train part** (DS3 web photos with a
known problem / healthy label; never scored, never trained on by today's models) and pick the threshold with the best
balanced accuracy (Youden's J). Then copy the model's predictions into predictions/<model>_cal.csv with the flags
recomputed at that threshold. score.py reports <model>_cal next to the original, on the bench part only, so the
bench photos never influence the threshold.

Models trained on the MetaDent train part later (v1.3) must NOT be calibrated here: they pick their threshold on
their own held-out slice when they're trained.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import MODELS  # noqa: E402
from common import CLASSES, MANIFEST, PRED_DIR, RESULTS  # noqa: E402

CAL_FILE = RESULTS / "calibration.json"


def youden(y, s):
    """threshold with the best balanced accuracy; ties -> the middle of the best run of cut points."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    cuts = np.unique(s)
    best, picks = -1.0, []
    for c in cuts:
        f = s >= c
        b = ((f[y == 1]).mean() + (~f[y == 0]).mean()) / 2
        if b > best + 1e-12:
            best, picks = b, [c]
        elif abs(b - best) <= 1e-12:
            picks.append(c)
    return float(np.median(picks)), float(best)


def main():
    man = pd.read_csv(MANIFEST, low_memory=False)
    cal = man[(man.dataset == "metadent") & (man.role == "train") & man.problem.notna()][["id", "problem"]]
    print(f"calibration photos: {len(cal)} ({int(cal.problem.sum())} problem / {int((cal.problem == 0).sum())} healthy)")
    out = {}
    for name, spec in MODELS.items():
        if spec.get("derived_from") or spec.get("trained_on_external"):
            continue
        f = PRED_DIR / f"{name}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        d = cal.merge(p[["id", "p_problem"]], on="id").dropna()
        if d.empty:
            print(f"{name}: no photo-level score, skipped")
            continue
        thr, b = youden(d.problem, d.p_problem)
        old = spec.get("conf")
        if old is None:     # classifier: its own threshold from the checkpoint
            flags = p.flag_problem.astype(str).str.lower().eq("true")
            pos = p.p_problem[flags]
            old = float(pos.min()) if len(pos) else None
        b_old = None
        if old is not None:
            fo = d.p_problem >= old
            b_old = float(((fo[d.problem == 1]).mean() + (~fo[d.problem == 0]).mean()) / 2)
        out[name] = dict(threshold=thr, calib_bal_acc=b, old_threshold=old, old_calib_bal_acc=b_old,
                         calib_auroc=float(roc_auc_score(d.problem, d.p_problem)), n=len(d))
        q = p.copy()
        q["flag_problem"] = q.p_problem >= thr
        is_det = q.boxes.notna().any()
        for c in CLASSES:   # detectors: per-condition flags at the same new cut; classifiers keep their argmax flags
            if is_det and q[f"p_{c}"].notna().any():
                q[f"flag_{c}"] = q[f"p_{c}"] >= thr
        q.to_csv(PRED_DIR / f"{name}_cal.csv", index=False)
        print(f"{name}: threshold {old if old is None else round(old, 3)} -> {thr:.3f} | balanced acc on the "
              f"calibration photos {b_old if b_old is None else round(b_old, 3)} -> {b:.3f}")
    RESULTS.mkdir(exist_ok=True)
    CAL_FILE.write_text(json.dumps(out, indent=1))
    print("wrote", CAL_FILE)


if __name__ == "__main__":
    main()
