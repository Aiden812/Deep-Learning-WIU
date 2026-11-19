"""Proves the adapters preprocess like the training notebooks: rerun Brandon's models through the adapters on
their own v1.2 test_fair photos and compare with the numbers the notebooks saved.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/sanity_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import load  # noqa: E402
from common import REPO, WORKSPACE  # noqa: E402

POOL = WORKSPACE / "datasets/oral_datasetv1.2/pool"


def bal_acc(y, f):
    return np.mean([(f[y == 1] == 1).mean(), (f[y == 0] == 0).mean()])


def run(name, paths, crop_cache):
    m = load(name)
    out = []
    for i in range(0, len(paths), 32):
        ims = []
        for p in paths[i:i + 32]:
            with Image.open(crop_cache / Path(p).name if crop_cache else POOL / p) as im:
                ims.append(im.convert("RGB"))
        out += m.predict_batch(ims)
    return np.array([r["p_problem"] for r in out]), np.array([r["flag_problem"] for r in out]).astype(int)


def main():
    pool = pd.read_csv(POOL / "pool.csv").set_index("image")
    fair = [p.removeprefix("./") for p in (POOL / "test_fair.txt").read_text().split()]
    y = pool.loc[fair, "has_issue"].to_numpy()
    res = REPO / "models"
    for name, metr, key, cache in [
            ("brandon_cls_v12", res / "classifier/results/Brandon/B2_effb0_issue_v12_metrics.json", "test_fair",
             WORKSPACE / "datasets/oral_datasetv1.2/cache_288_nopad"),
            ("brandon_yolo_v12", res / "yolo_caries/results/Brandon/Y2_yolo11n_640_v12_metrics.json", None, None)]:
        p, f = run(name, fair, cache)
        saved = json.loads(metr.read_text())
        saved = saved[key] if key else saved["image_level"]["test_fair"]
        print(f"{name}: n={len(y)}  AUROC {roc_auc_score(y, p):.4f} (notebook {saved['roc_auc']:.4f})  "
              f"bal acc {bal_acc(y, f):.4f} (notebook {saved['balanced_accuracy']:.4f})")


if __name__ == "__main__":
    main()
