"""Example pictures for the report: dentist boxes vs each detector's boxes on the same benchmark photo.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/examples.py            # 4 AlphaDent + 3 Autooral
    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/examples.py --n 8 --seed 7

Photos are picked at random (seeded) from the bench part, not cherry-picked. Each model's boxes are shown at its
own operating confidence (the same cut used for recall/precision in the report). Needs run.py to have been run.
Output: benchmark/results/examples/<dataset>_<photo>.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, MANIFEST, OUT, PRED_DIR, REPO, RESULTS  # noqa: E402
from adapters import MODELS  # noqa: E402

DETECTORS = ["aiden_yolo11s", "brandon_yolo_v12", "brandon_yolo_v11", "daniel_yolo", "junwei_yolo11n"]
NICE = {"aiden_yolo11s": "Aiden YOLO11s", "brandon_yolo_v12": "Brandon YOLO v12", "brandon_yolo_v11": "Brandon YOLO v11",
        "daniel_yolo": "Daniel YOLO12m", "junwei_yolo11n": "JunWei YOLO11n"}
COLORS = {"Calculus": "#f2c14e", "Caries": "#e4572e", "Hypodontia": "#a05fd6", "Mouth_Ulcer": "#ff4fa3",
          "Tooth_Discoloration": "#29a3e0"}
TRUTH = "#2ecc40"
WHAT = {"alphadent": "AlphaDent: dentist outlined each caries lesion",
        "autooral": "Autooral: dentist outlined the ulcer"}


def op_conf(name):
    side = (REPO / MODELS[name]["weights"]).with_suffix(".json")
    if side.exists():
        c = json.loads(side.read_text()).get("conf")
        if c is not None:
            return float(c)
    return float(MODELS[name].get("conf", 0.25))


def truth_boxes(label_rel):
    p = OUT / label_rel
    if not isinstance(label_rel, str) or not p.exists():
        return []
    rows = [list(map(float, l.split())) for l in p.read_text().splitlines() if l.strip()]
    return [(int(c), x - w / 2, y - h / 2, x + w / 2, y + h / 2) for c, x, y, w, h in rows]


def draw(ax, im, boxes, color_of, label_of, lw):
    W, H = im.size
    ax.imshow(im)
    for b in boxes:
        c, x1, y1, x2, y2 = b[0], *b[-4:]
        ax.add_patch(mpatches.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False,
                                        edgecolor=color_of(b), linewidth=lw))
        lab = label_of(b)
        if lab:
            ax.text(x1 * W, y1 * H, lab, fontsize=6, color="white", va="bottom",
                    bbox=dict(facecolor=color_of(b), edgecolor="none", pad=0.8, alpha=0.85))
    ax.set_xticks([]), ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="photos per dataset (default 4 AlphaDent, 3 Autooral)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    man = pd.read_csv(MANIFEST)
    bench = man[(man.role == "bench") & man.dataset.isin(["alphadent", "autooral"]) & (man.n_boxes > 0)]
    picks = pd.concat([bench[bench.dataset == "alphadent"].sample(a.n or 4, random_state=a.seed),
                       bench[bench.dataset == "autooral"].sample(a.n or 3, random_state=a.seed)])

    preds, confs = {}, {}
    for m in DETECTORS:
        f = PRED_DIR / f"{m}.csv"
        if f.exists():
            preds[m] = pd.read_csv(f, usecols=["id", "boxes"]).set_index("id")["boxes"]
            confs[m] = op_conf(m)
    if not preds:
        sys.exit("no detector predictions found: run run.py first")

    out = RESULTS / "examples"
    out.mkdir(parents=True, exist_ok=True)
    for _, r in picks.iterrows():
        im = Image.open(OUT / r.image).convert("RGB")
        im.thumbnail((1280, 1280))
        panels = 1 + len(preds)
        cols = 3
        rows = -(-panels // cols)
        aspect = im.size[1] / im.size[0]
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2 * aspect + 0.9))
        axes = axes.ravel()

        gt = truth_boxes(r.label)
        draw(axes[0], im, gt, lambda b: TRUTH, lambda b: None, 1.6)
        axes[0].set_title(f"Dentist (truth): {len(gt)} box{'es' if len(gt) != 1 else ''}", fontsize=9,
                          color=TRUTH, fontweight="bold")
        for ax, (m, s) in zip(axes[1:], preds.items()):
            raw = json.loads(s.get(r.id, "[]") or "[]") if isinstance(s.get(r.id), str) else []
            keep = [b for b in raw if b[1] >= confs[m]]
            draw(ax, im, gt, lambda b: TRUTH, lambda b: None, 0.8)          # faint truth underneath
            for p in ax.patches:
                p.set_alpha(0.45), p.set_linestyle("--")
            draw(ax, im, keep, lambda b: COLORS[CLASSES[b[0]]],
                 lambda b: f"{CLASSES[b[0]].replace('_', ' ')} {b[1]:.2f}", 1.6)
            ax.set_title(f"{NICE[m]}: {len(keep)} box{'es' if len(keep) != 1 else ''} (conf >= {confs[m]:.2f})",
                         fontsize=9)
        for ax in axes[panels:]:
            ax.axis("off")

        handles = [mpatches.Patch(color=TRUTH, label="dentist box (dashed under model panels)")] + \
                  [mpatches.Patch(color=COLORS[c], label=c.replace("_", " ")) for c in CLASSES]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False)
        fig.suptitle(f"{WHAT[r.dataset]}  |  photo {r.id.split('/')[-1]} (random bench photo, no model trained on it)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        dst = out / f"{r.id.replace('/', '_')}.jpg"
        fig.savefig(dst, dpi=110, pil_kwargs=dict(quality=85))
        plt.close(fig)
        print(dst.relative_to(REPO))


if __name__ == "__main__":
    main()
