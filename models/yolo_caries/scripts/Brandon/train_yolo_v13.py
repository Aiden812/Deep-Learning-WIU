"""Fine-tune Brandon's YOLO v1.2 detector on the v1.3 set (build_yolo_v13.py), then pick its operating confidence.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/models/yolo_caries/scripts/Brandon/train_yolo_v13.py
    YOLO_SMOKE=1 ...   (2% of the data, 1 epoch: checks every step runs)

Re-running resumes from last.pt (the laptop overheats on long runs). At the end:
  * best.pt -> models/yolo_caries/weights/Brandon/yolo_detector_v13.pt
  * conf picked on the held-out external val slice (MetaDent photos with a problem / healthy label, best balanced
    accuracy) -> yolo_detector_v13.json sidecar, which the benchmark adapter reads
  * box metrics on the external val slice and on v1.2 test / test_fair_human -> results/Brandon/Y4_..._metrics.json
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))      # Deep-Learning-WIU/models
from v13_split import REPO, WORKSPACE, external_train_rows, keep_awake  # noqa: E402

SMOKE = os.environ.get("YOLO_SMOKE") == "1"
WORK = WORKSPACE / "datasets" / "yolo_work_v1.3"
START = REPO / "models/yolo_caries/weights/Brandon/yolo_detector_v12.pt"
EXPERIMENT = "Y4_yolo11n_960_v13" + ("_smoke" if SMOKE else "")
IMGSZ = 960
BATCH = int(os.environ.get("YOLO_BATCH", 8))
EPOCHS = 1 if SMOKE else int(os.environ.get("YOLO_EPOCHS", 15))
PATIENCE = 6
WORKERS = 6
SEED = 42
OUT_DIR = REPO / "models/yolo_caries/results/Brandon"
WEIGHTS_OUT = REPO / "models/yolo_caries/weights/Brandon" / ("yolo_detector_v13" + ("_smoke" if SMOKE else "") + ".pt")
NAMES = ["Calculus", "Caries", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]


def youden(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    best, picks = -1, []
    for c in np.unique(s):
        f = s >= c
        b = (f[y == 1].mean() + (~f[y == 0]).mean()) / 2
        if b > best + 1e-12:
            best, picks = b, [c]
        elif abs(b - best) <= 1e-12:
            picks.append(c)
    return float(np.median(picks)), float(best)


def train():
    from ultralytics import YOLO
    run_dir = WORK / "runs" / EXPERIMENT
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("epoch", -1) >= 0:
            print(f"resuming after epoch {ck['epoch'] + 1}", flush=True)
            YOLO(str(last)).train(resume=True)
            return run_dir
        print("run already finished; using its weights", flush=True)
        return run_dir
    model = YOLO(str(START))
    model.train(data=str(WORK / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, workers=WORKERS,
                patience=PATIENCE, seed=SEED, cos_lr=True, optimizer="AdamW", lr0=5e-4, warmup_epochs=1,
                close_mosaic=3, fraction=0.02 if SMOKE else 1.0, project=str(WORK / "runs"), name=EXPERIMENT,
                exist_ok=True, plots=True, verbose=False, amp=True)
    return run_dir


def main():
    keep_awake()
    from ultralytics import YOLO
    t0 = time.time()
    run_dir = train()
    best_pt = run_dir / "weights" / "best.pt"
    print(f"training done ({(time.time() - t0) / 60:.0f} min this session) -> {best_pt}", flush=True)
    best = YOLO(str(best_pt))

    # ---- operating confidence on the held-out external slice -----------------------------------------------------
    ext = external_train_rows()
    cal = ext[(ext.role13 == "val") & (ext.dataset == "metadent") & ext.problem.notna()]
    rows = pd.read_csv(WORK / "external_rows.csv")
    rows = rows[(rows.rep == 0) & (rows.role13 == "val")]
    rows["id"] = rows.path.map(lambda p: Path(p).stem.replace("__", "/", 1))
    cal = cal.merge(rows[["id", "path"]], on="id")
    tops = []
    for i in range(0, len(cal), 32):
        for r in best.predict(cal.path.iloc[i:i + 32].tolist(), imgsz=IMGSZ, conf=0.001, verbose=False):
            tops.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    cal["top"] = tops
    from sklearn.metrics import roc_auc_score
    conf, bal = youden(cal.problem, cal.top)
    conf = max(conf, 0.01)
    auc = float(roc_auc_score(cal.problem, cal.top))
    print(f"operating conf {conf:.3f} (balanced acc {bal:.3f}, AUROC {auc:.3f} on {len(cal)} held-out MetaDent photos)")

    # ---- box metrics ---------------------------------------------------------------------------------------------
    box = {}
    for name in ["val_external", "v12_test_fair_human", "v12_test_fair"]:
        y = WORK / f"eval_{name}.yaml"
        if not y.exists():
            continue
        m = best.val(data=str(y), split="val", imgsz=IMGSZ, batch=BATCH, workers=WORKERS, plots=False,
                     project=str(WORK / "runs"), name=f"{EXPERIMENT}_{name}", exist_ok=True, verbose=False)
        box[name] = dict(mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
                         per_class_AP50={NAMES[int(c)]: float(m.box.ap50[i]) for i, c in enumerate(m.box.ap_class_index)})
        print(name, json.dumps(box[name]), flush=True)

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_pt, WEIGHTS_OUT)
    WEIGHTS_OUT.with_suffix(".json").write_text(json.dumps(dict(
        conf=conf, imgsz=IMGSZ, picked_on="held-out 20% of the MetaDent train part (benchmark manifest), best balanced accuracy",
        calib_bal_acc=bal, calib_auroc=auc, n_calib=len(cal)), indent=1))
    hist = pd.read_csv(run_dir / "results.csv")
    hist.columns = hist.columns.str.strip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{EXPERIMENT}_metrics.json").write_text(json.dumps(dict(
        experiment=EXPERIMENT, start_weights=str(START.relative_to(REPO)), imgsz=IMGSZ, batch=BATCH,
        epochs_run=len(hist), data=json.loads((WORK / "summary.json").read_text()), conf=conf,
        calib=dict(bal_acc=bal, auroc=auc, n=len(cal)), box=box), indent=1))
    hist.to_csv(OUT_DIR / f"{EXPERIMENT}_history.csv", index=False)
    for f in ("results.png", "BoxPR_curve.png", "confusion_matrix_normalized.png"):
        if (run_dir / f).exists():
            (OUT_DIR / "figures").mkdir(exist_ok=True)
            shutil.copyfile(run_dir / f, OUT_DIR / "figures" / f"{EXPERIMENT}_{f}")
    print("saved", WEIGHTS_OUT, flush=True)


if __name__ == "__main__":
    main()
