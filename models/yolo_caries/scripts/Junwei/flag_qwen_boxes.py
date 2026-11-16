"""Check Qwen-drawn boxes in Daniel's oral_dataset with a model that never trained on them, and write a cleaned train list.

A Qwen box is confirmed when the model predicts the same class over it: the prediction covers at least half of the
Qwen box and is at most 10x its size (Qwen boxes are about a third the size of human ones, so plain IoU would reject
good matches). A photo is bad when fewer than half of its Qwen boxes are confirmed.

Writes, next to --out:
    <out>/qwen_check.csv   one row per Qwen-boxed train photo: image, qwen_boxes, confirmed, bad
    <out>/train.txt        Daniel's yolo_train.txt minus the bad photos (absolute paths)
    <out>.yaml             YOLO config: that train list, Daniel's val/test

Usage:
    python models/yolo_caries/scripts/Junwei/flag_qwen_boxes.py --weights <best.pt of a run without Qwen photos>
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "data/raw/oral_dataset"
CARIES = "1"


def qwen_boxes(row: dict) -> list[list[float]]:
    """Boxes in this photo that Qwen drew, as [class, cx, cy, w, h]."""
    lab = DATA / "labels" / row["split"] / (Path(row["image"]).stem + ".txt")
    boxes = [l.split() for l in lab.read_text().splitlines() if l.strip()]
    if row["boxes_by"] == "model":
        mine = boxes  # every box here is Qwen's; healthy photos have none
    elif row["boxes_by"].startswith("human (caries)"):
        mine = [b for b in boxes if b[0] != CARIES]  # people drew caries; Qwen drew the other classes
    else:
        mine = []
    return [[float(v) for v in b] for b in mine]


def confirmed(q: list[float], preds: list[list[float]]) -> bool:
    """True if a predicted box of the same class covers at least half of q and is at most 10x its area."""
    c, x, y, w, h = q
    for pc, px, py, pw, ph in preds:
        if int(pc) != int(c) or pw * ph > 10 * w * h:
            continue
        ix = max(0.0, min(x + w / 2, px + pw / 2) - max(x - w / 2, px - pw / 2))
        iy = max(0.0, min(y + h / 2, py + ph / 2) - max(y - h / 2, py - ph / 2))
        if ix * iy >= 0.5 * w * h:
            return True
    return False


assert confirmed([1, .5, .5, .1, .1], [[1, .5, .5, .2, .2]])        # small Qwen box inside a bigger prediction
assert not confirmed([1, .5, .5, .1, .1], [[0, .5, .5, .1, .1]])    # wrong class
assert not confirmed([1, .5, .5, .1, .1], [[1, .8, .8, .1, .1]])    # elsewhere
assert not confirmed([1, .5, .5, .1, .1], [[1, .5, .5, .9, .9]])    # one huge box does not confirm everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO / "data/processed/oral5_qwen_checked")
    parser.add_argument("--conf", type=float, default=0.1)  # low on purpose: a weak agreeing box still counts
    args = parser.parse_args()

    rows = {r["image"]: r for r in csv.DictReader(open(DATA / "labels.csv", encoding="utf-8"))}
    train = [l.strip()[2:] for l in (DATA / "yolo_train.txt").read_text().splitlines() if l.strip()]
    to_check = {img: qwen_boxes(rows[img]) for img in train}
    to_check = {img: b for img, b in to_check.items() if b}

    model = YOLO(args.weights)
    results = []
    for img, qb in to_check.items():
        r = model.predict(DATA / img, conf=args.conf, verbose=False)[0]
        preds = [[c, *xywh] for c, xywh in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        ok = sum(confirmed(q, preds) for q in qb)
        results.append({"image": img, "qwen_boxes": len(qb), "confirmed": ok, "bad": int(ok < len(qb) / 2)})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "qwen_check.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "qwen_boxes", "confirmed", "bad"])
        writer.writeheader()
        writer.writerows(results)
    bad = {r["image"] for r in results if r["bad"]}
    keep = [img for img in train if img not in bad]
    (args.out / "train.txt").write_text("".join(f"{DATA / img}\n" for img in keep))
    args.out.with_suffix(".yaml").write_text(
        f"# Daniel's oral_dataset, minus train photos whose Qwen boxes a model could not confirm. val/test unchanged.\n"
        f"path: {DATA.as_posix()}\ntrain: {(args.out / 'train.txt').as_posix()}\nval: images/val\ntest: images/test\n"
        f"names: [Calculus, Caries, Hypodontia, Mouth_Ulcer, Tooth_Discoloration]\n")

    n_box, n_ok = sum(r["qwen_boxes"] for r in results), sum(r["confirmed"] for r in results)
    print(f"RESULT checked {len(results)} photos, {n_box} Qwen boxes: {n_ok} confirmed ({n_ok / n_box:.0%})")
    print(f"RESULT bad photos: {len(bad)} ({len(bad) / len(results):.0%}) -> train {len(train)} -> {len(keep)}")
    by_source = {}
    for r in results:
        d = rows[r["image"]]["dataset"]
        by_source.setdefault(d, [0, 0])
        by_source[d][0] += 1
        by_source[d][1] += r["bad"]
    for d, (n, b) in sorted(by_source.items()):
        print(f"RESULT   {d}: {b}/{n} photos bad ({b / n:.0%})")
