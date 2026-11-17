"""Step 4: run models on every usable benchmark photo and save their raw outputs.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/run.py brandon_cls_v12 brandon_yolo_v12
    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/run.py all            # every model whose weights exist
    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/run.py all --speed    # CPU ms/photo on 50 photos
    Deep-Learning-WIU/.venv/Scripts/python.exe Deep-Learning-WIU/benchmark/run.py team_keras_6class   # Keras model

Predicts on bench AND train rows (the train part is unseen by today's models, so it gives extra "all photos" numbers;
score.py reports the bench part first). Writes data/processed/benchmark/predictions/<model>.csv and appends as it
goes, so a run stopped by the laptop overheating resumes where it stopped. Uses only the standard library + numpy +
PIL here, so it also runs in the repo's TensorFlow venv.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--weights", help="override the weights path (one model only)")
ap.add_argument("--speed", action="store_true", help="time CPU inference, batch 1, on 50 bench photos")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()
if args.speed:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""     # before torch/tf are imported: everything runs on the CPU

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import MODELS, load  # noqa: E402
from common import CLASSES, MANIFEST, OUT, PRED_DIR, REPO  # noqa: E402

FIELDS = ["id", "p_problem", "flag_problem", *[f"p_{c}" for c in CLASSES], *[f"flag_{c}" for c in CLASSES], "boxes"]


def open_rgb(rel):
    with Image.open(OUT / rel) as im:
        return im.convert("RGB")


def flatten(i, r):
    row = dict(id=i, p_problem=r["p_problem"], flag_problem=r["flag_problem"])
    for c in CLASSES:
        row[f"p_{c}"] = r["p"].get(c)
        row[f"flag_{c}"] = r["flag"].get(c)
    row["boxes"] = None if r["boxes"] is None else json.dumps(r["boxes"], separators=(",", ":"))
    return {k: ("" if v is None else v) for k, v in row.items()}


def run_model(name, man):
    out_f = PRED_DIR / f"{name}.csv"
    done = set()
    if out_f.exists():
        with open(out_f, newline="", encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    todo = [r for r in man if r["id"] not in done]
    if not todo:
        print(f"{name}: all {len(man)} photos already predicted")
        return
    m = load(name, args.weights)
    print(f"{name}: predicting {len(todo)} photos ({len(done)} done before)", flush=True)
    t0 = time.time()
    new = not out_f.exists()
    with open(out_f, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for s in range(0, len(todo), args.batch):
            chunk = todo[s:s + args.batch]
            res = m.predict_batch([open_rgb(r["image"]) for r in chunk])
            w.writerows(flatten(r["id"], x) for r, x in zip(chunk, res))
            f.flush()
            if (s // args.batch) % 25 == 0:
                print(f"  {s + len(chunk)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"{name}: done in {time.time() - t0:.0f}s")


def speed(name, man):
    m = load(name, args.weights)
    bench = [r for r in man if r["role"] == "bench"]
    random.Random(0).shuffle(bench)
    ims = [open_rgb(r["image"]) for r in bench[:55]]
    for im in ims[:5]:
        m.predict_batch([im])                      # warm-up
    ts = []
    for im in ims[5:]:
        t = time.perf_counter(); m.predict_batch([im]); ts.append((time.perf_counter() - t) * 1000)
    w = Path(args.weights) if args.weights else REPO / MODELS[name]["weights"]
    row = dict(model=name, cpu_ms_median=round(float(np.median(ts)), 1), cpu_ms_p90=round(float(np.percentile(ts, 90)), 1),
               size_mb=round(w.stat().st_size / 1e6, 2), n=len(ts))
    f = PRED_DIR / "speed.csv"
    rows = []
    if f.exists():
        with open(f, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["model"] != name]
    with open(f, "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(row))
        wr.writeheader(); wr.writerows(rows + [row])
    print(row)


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        man = [r for r in csv.DictReader(f) if r["role"] in ("bench", "train")]
    names = list(MODELS) if args.models == ["all"] else args.models
    in_tf_venv = "tensorflow" in {p.name for p in Path(sys.prefix, "Lib", "site-packages").glob("tensorflow")}
    for name in names:
        w = Path(args.weights) if args.weights else REPO / MODELS[name]["weights"]
        if args.models == ["all"] and (MODELS[name].get("python") == "repo_venv") != in_tf_venv:
            print(f"{name}: skipped in this Python (see the docstring for which venv runs it)")
            continue
        if not w.exists():
            print(f"{name}: skipped, no weights at {w}")
            continue
        (speed if args.speed else run_model)(name, man)


if __name__ == "__main__":
    main()
