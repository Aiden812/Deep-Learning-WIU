"""RF-DETR training, prediction and model size for the stage-2 oral detector notebook.

RF-DETR (Roboflow's DINOv2-backbone DETR) needs transformers 5.x, while other code in this repo (the demo's CLIP mouth
gate) runs on transformers 4.x. RF-DETR is therefore installed into <repo>/.rfdetr_site, and this script runs in its
own process with that folder first on the import path:

    py -3.13 -m pip install --no-deps --target .rfdetr_site "rfdetr[train]==1.12.1" ...   (see the notebook's setup)
    PYTHONPATH=.rfdetr_site py -3.13 models/yolo_caries/scripts/Daniel/rfdetr_runner.py <command> ...

The notebook models/yolo_caries/notebooks/Daniel/yolo_oral_detector.ipynb calls it through subprocess.

Commands:
  train    fine-tune a COCO-pretrained RF-DETR on the notebook's training copy (YOLO label files), writing
           checkpoint_best_total.pth and an Ultralytics-style results.csv to --out
  predict  every detection with confidence >= --conf for a list of images, as CSV
           (image, cls, conf, x1, y1, x2, y2; box corners as fractions of the image)
  info     parameters (M), GFLOPs and GPU forward time per image of a checkpoint, as JSON
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def variant_class(name):
    # Imported lazily so `--help` works without the RF-DETR install.
    import rfdetr
    return {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
            "base": rfdetr.RFDETRBase, "large": rfdetr.RFDETRLarge}[name]


def link(src, dst):
    # Hard link (no extra disk space), or a copy when the two paths are on different volumes.
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_dataset(data_root, train_list, out):
    """Build the folder layout RF-DETR reads from the notebook's training copy.

    RF-DETR reads YOLO labels, but only from image *folders* (images/<split> + labels/<split>), while the notebook's
    data.yaml names the training images in a .txt list. This writes <out>/images|labels/{train,val} as hard links of
    the images in `train_list` and of every val image, plus a data.yaml. Files no longer wanted are removed, so the
    folder always matches the list. Returns the number of images per split.
    """
    names = yaml.safe_load(open(data_root / "data.yaml"))["names"]
    lines = [ln.strip().removeprefix("./") for ln in (data_root / train_list).read_text().splitlines() if ln.strip()]
    want = {"train": [Path(p) for p in lines],
            "val": [Path("images", "val", p.name) for p in (data_root / "images" / "val").iterdir()]}
    counts = {}
    for split, imgs in want.items():
        for kind in ("images", "labels"):
            (out / kind / split).mkdir(parents=True, exist_ok=True)
        keep = {p.name for p in imgs}
        keep_lbl = {Path(n).with_suffix(".txt").name for n in keep}
        for p in (out / "images" / split).iterdir():
            if p.name not in keep:
                p.unlink()
        for p in (out / "labels" / split).iterdir():
            if p.name not in keep_lbl:
                p.unlink()
        for img in imgs:
            lbl = Path("labels", *img.parts[1:]).with_suffix(".txt")
            dst_img, dst_lbl = out / "images" / split / img.name, out / "labels" / split / lbl.name
            if not dst_img.exists():
                link(data_root / img, dst_img)
            if not dst_lbl.exists():
                link(data_root / lbl, dst_lbl)
        counts[split] = len(imgs)
    (out / "data.yaml").write_text(yaml.safe_dump(dict(train="images/train", val="images/val", nc=len(names),
                                                       names=names), sort_keys=False, default_flow_style=None))
    return counts


def to_results_csv(metrics_csv, out_csv):
    """Convert RF-DETR's Lightning metrics.csv to the column names of an Ultralytics results.csv.

    The notebook's comparison cell reads epoch, time, train/val box and class losses and val mAP from results.csv.
    Lightning writes training-step and validation-epoch rows separately, so they are averaged per epoch here.
    RF-DETR's box loss is L1 + GIoU, and its class loss is a focal-style varifocal loss: the curves are comparable in
    shape with the YOLO runs, not in absolute value. Columns RF-DETR does not log are left empty.
    """
    m = pd.read_csv(metrics_csv)
    m = m[m.epoch.notna()].copy()
    m["epoch"] = m.epoch.astype(int)
    per_epoch = m.groupby("epoch").mean(numeric_only=True)

    def pick(*cands):
        for c in cands:
            if c in per_epoch.columns and per_epoch[c].notna().any():
                return per_epoch[c]
        return pd.Series(np.nan, index=per_epoch.index)

    def summed(prefix, parts):
        cols = [f"{prefix}/{p}" for p in parts if f"{prefix}/{p}" in per_epoch.columns]
        return per_epoch[cols].sum(axis=1, min_count=1) if cols else pd.Series(np.nan, index=per_epoch.index)

    res = pd.DataFrame({
        "epoch": per_epoch.index + 1,                       # Ultralytics counts epochs from 1
        "time": pick("time", "train/epoch_time"),
        "train/box_loss": summed("train", ["loss_bbox", "loss_giou"]),
        "train/cls_loss": pick("train/loss_ce", "train/loss_vfl", "train/loss_cls"),
        "metrics/precision(B)": pick("val/precision"),
        "metrics/recall(B)": pick("val/recall", "val/mAR", "val/mar"),
        "metrics/mAP50(B)": pick("val/mAP_50", "val/map_50", "val/mAP50"),
        "metrics/mAP50-95(B)": pick("val/mAP_50_95", "val/map", "val/mAP"),
        "val/box_loss": summed("val", ["loss_bbox", "loss_giou"]),
        "val/cls_loss": pick("val/loss_ce", "val/loss_vfl", "val/loss_cls"),
    }).reset_index(drop=True)
    res = res[res["metrics/mAP50-95(B)"].notna()].reset_index(drop=True)
    res.to_csv(out_csv, index=False)
    return res


def fill_epoch_time(results_csv, seconds):
    # RF-DETR logs no wall-clock time per epoch. Spread this session's training time evenly over its epochs, so the
    # notebook's "min / epoch" column holds the average (cumulative seconds, as Ultralytics writes them).
    res = pd.read_csv(results_csv)
    res["time"] = (np.arange(len(res)) + 1) * seconds / max(len(res), 1)
    res.to_csv(results_csv, index=False)


def cmd_train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    counts = build_dataset(Path(a.data_root), a.train_list, Path(a.dataset_dir))
    print(f"[rfdetr] dataset {a.dataset_dir}: {counts} ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()

    last = out / "last.ckpt"
    model = variant_class(a.variant)()
    kwargs = dict(
        dataset_dir=a.dataset_dir, dataset_file="yolo", output_dir=str(out), epochs=a.epochs,
        batch_size=a.batch, grad_accum_steps=a.grad_accum, resolution=a.imgsz, num_workers=a.workers,
        seed=a.seed, lr=a.lr, lr_encoder=a.lr_encoder, weight_decay=a.weight_decay,
        early_stopping=a.patience > 0, early_stopping_patience=max(a.patience, 1),
        tensorboard=False, progress_bar="tqdm", device="cuda", run_test=False,
    )
    if last.exists():
        kwargs["resume"] = str(last)
        print(f"[rfdetr] resuming from {last}", flush=True)
    model.train(**kwargs)

    best = out / "checkpoint_best_total.pth"
    assert best.exists(), f"training finished without {best}"
    if (out / "metrics.csv").exists():
        to_results_csv(out / "metrics.csv", out / "results.csv")
        if "resume" not in kwargs:                         # after a resume, the session time misses earlier epochs
            fill_epoch_time(out / "results.csv", time.time() - t0)
    (out / "done.json").write_text(json.dumps(dict(best=str(best), minutes=(time.time() - t0) / 60, args=vars(a)),
                                              indent=2))
    print(f"[rfdetr] done in {(time.time() - t0) / 60:.1f} min -> {best}", flush=True)


def load(ckpt):
    import rfdetr
    model = rfdetr.RFDETR.from_checkpoint(ckpt)
    return model


def cmd_predict(a):
    model = load(a.ckpt)
    root = Path(a.image_root)
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    cls_of = {n: i for i, n in enumerate(names)}        # map by NAME: RF-DETR also emits a no-object slot (id = nc)
    images = [ln.strip() for ln in Path(a.images).read_text().splitlines() if ln.strip()]
    recs = []
    for i in range(0, len(images), a.batch):
        chunk = images[i:i + a.batch]
        pil = [Image.open(root / p).convert("RGB") for p in chunk]
        dets = model.predict(pil, threshold=a.conf)
        if not isinstance(dets, list):
            dets = [dets]
        for img, im, d in zip(chunk, pil, dets):
            w, h = im.size
            for n, p, (x1, y1, x2, y2) in zip(d.data["class_name"].tolist(), d.confidence.tolist(), d.xyxy.tolist()):
                if n in cls_of:
                    recs.append((img, cls_of[n], float(p), x1 / w, y1 / h, x2 / w, y2 / h))
        if (i // a.batch) % 20 == 0:
            print(f"[rfdetr] predict {min(i + a.batch, len(images))}/{len(images)}", flush=True)
    pd.DataFrame(recs, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(a.out, index=False)
    print(f"[rfdetr] {len(recs)} boxes on {len(images)} images -> {a.out}", flush=True)


def torch_module(model):
    # The plain nn.Module inside the RF-DETR wrapper (its location differs between rfdetr versions).
    for path in ["model.model", "model"]:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
        if isinstance(obj, torch.nn.Module):
            return obj
    raise AttributeError("could not find the torch module inside the RF-DETR model")


def cmd_info(a):
    from torch.utils.flop_counter import FlopCounterMode
    model = load(a.ckpt)
    net = torch_module(model).cuda().eval()
    for p in net.parameters():                             # FlopCounterMode fails under no_grad on this model
        p.requires_grad_(False)
    res = int(model.model_config.resolution)
    params = sum(p.numel() for p in net.parameters())
    with FlopCounterMode(display=False) as fc:
        net(torch.zeros(1, 3, res, res, device="cuda"))
    x = torch.rand(a.batch, 3, res, res, device="cuda")
    with torch.no_grad():                                  # FP32 batched forward, as Ultralytics' val speed
        for _ in range(5):
            net(x)
        torch.cuda.synchronize()
        t0, n = time.perf_counter(), 10
        for _ in range(n):
            net(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n / a.batch * 1000
    info = dict(params_M=params / 1e6, GFLOPs=fc.get_total_flops() / 1e9, inference_ms=ms, resolution=res)
    Path(a.out).write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--variant", default="medium", choices=["nano", "small", "medium", "base", "large"])
    t.add_argument("--data-root", required=True, help="the notebook's training copy (holds data.yaml)")
    t.add_argument("--train-list", default="yolo_train.txt", help="image list inside --data-root")
    t.add_argument("--dataset-dir", required=True, help="where to build the RF-DETR folder layout (hard links)")
    t.add_argument("--out", required=True, help="run folder: checkpoints, metrics.csv, results.csv")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--patience", type=int, default=10, help="early stopping on val mAP; 0 = off")
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch x grad-accum")
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lr-encoder", type=float, default=1.5e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    pr = sub.add_parser("predict")
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--image-root", required=True)
    pr.add_argument("--images", required=True, help="text file, one image path per line relative to --image-root")
    pr.add_argument("--out", required=True)
    pr.add_argument("--conf", type=float, default=0.001)
    pr.add_argument("--batch", type=int, default=16)
    i = sub.add_parser("info")
    i.add_argument("--ckpt", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    {"train": cmd_train, "predict": cmd_predict, "info": cmd_info}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
