"""Shared by the v1.3 training scripts (YOLO and classifier): which external photos they may train on.

Only rows with role == "train" in benchmark/manifest.csv are ever used (the bench part stays unseen). Those are
split once more, by group (patient / near-duplicate cluster), into
    fit  (80%) -> trained on
    val  (20%) -> picks the best epoch and the operating threshold (never trained on)
The split is a hash of the group name, so it is the same every run and on every laptop.
"""
import hashlib
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
MANIFEST = REPO / "benchmark" / "manifest.csv"
BENCH_OUT = REPO / "data" / "processed" / "benchmark"      # manifest image / label paths are relative to this
RAW = REPO / "data" / "raw"
POOL = WORKSPACE / "datasets" / "oral_datasetv1.2" / "pool"
VAL_SHARE = 20     # percent of groups


def role13(group: str) -> str:
    h = int(hashlib.md5(f"v13|{group}".encode()).hexdigest(), 16) % 100
    return "val" if h < VAL_SHARE else "fit"


def external_train_rows() -> pd.DataFrame:
    m = pd.read_csv(MANIFEST, low_memory=False)
    t = m[m.role == "train"].copy()
    assert not (t.role == "bench").any()
    t["role13"] = t.group.astype(str).map(role13)
    return t.reset_index(drop=True)


def keep_awake():
    """Stop Windows sleeping while training (lid-close still sleeps)."""
    import os
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
