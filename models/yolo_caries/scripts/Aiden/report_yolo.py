"""YOLO report and graphs (Aiden): rebuilt from models/yolo_caries/results/Aiden/<model>/metrics.json + results.csv.

    python models/yolo_caries/scripts/Aiden/report_yolo.py

Writes models/yolo_caries/results/Aiden/yolo_report.txt and figures/yolo_overview.png, figures/yolo_training_curves.png.
train_yolo.py calls write_all() after every run. Ultralytics' own plots (PR curves, confusion matrices) are in each
model's eval_* folders.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "models/yolo_caries/results/Aiden"
DATA = REPO / "data/raw/oral_dataset"
LEAKS = REPO / "data/processed/oral_dataset_leaks.csv"
# validated pair (dataviz checker: CVD dE 24.7, contrast >= 3:1); models are the series, in a fixed order
PALETTE = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID = "#1a1a19", "#5f5e5a", "#e4e3df"
FEW = 20
SETS = [("full", "Daniel's full test"), ("person", "Daniel's person-labelled test"), ("clean", "Clean test"),
        ("clean_person", "Clean + person-labelled test")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.dpi": 150, "legend.frameon": False})


def load():
    runs = []
    for m in sorted(RESULTS.glob("*/metrics.json")):
        r = json.loads(m.read_text())
        r["color"] = PALETTE[len(runs) % len(PALETTE)]
        h = pd.read_csv(m.parent / "results.csv")
        h.columns = h.columns.str.strip()
        r["hist"] = h
        runs.append(r)
    return runs


def clean_counts(classes):
    """Clean test images per class (how much each clean score can be trusted)."""
    d = pd.read_csv(DATA / "labels.csv", keep_default_na=False).merge(pd.read_csv(LEAKS)[["md5", "leaked"]], on="md5", how="left")
    d = d[(d.split == "test") & ~d.leaked.fillna(False).astype(bool)]
    return {c: int(d[c].sum()) for c in classes}


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def dots(ax, labels, values, runs, xlim=(0, 1.02)):
    y = np.arange(len(labels))[::-1]
    v = np.array(values, dtype=float)
    off = 0.12 * (np.arange(len(runs)) - (len(runs) - 1) / 2)
    for i in range(len(labels)):
        ax.plot(v[i], y[i] - off, color=GRID, linewidth=3, zorder=1, solid_capstyle="round")
    for j, r in enumerate(runs):
        ax.scatter(v[:, j], y - off[j], s=60, color=r["color"], edgecolor="white", linewidth=1.5, zorder=3)
    for i in range(len(labels)):
        if np.isnan(v[i]).all():
            continue
        j = int(np.nanargmax(v[i]))
        ax.annotate(f"{v[i, j]:.3f}", (v[i, j], y[i]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    tidy(ax, grid="x")


def legend(fig, runs):
    handles = [plt.Line2D([], [], color=r["color"], marker="o", linewidth=2, markersize=7) for r in runs]
    fig.legend(handles, [r["model"] for r in runs], loc="upper right", ncol=len(runs), fontsize=10)


def overview(runs, classes, n_clean):
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.9))
    rows = [(f"{label}: mAP50", k, "map50") for k, label in SETS] + [(f"{label}: mAP50-95", k, "map50_95") for k, label in SETS[::2]]
    dots(ax[0], [n for n, _, _ in rows], [[r["results"][k][m] for r in runs] for _, k, m in rows], runs)
    ax[0].set_title("Overall (higher is better)")
    get = lambda r, k, c: r["results"][k]["per_class"].get(c, {}).get("ap50", np.nan)
    dots(ax[1], classes, [[get(r, "full", c) for r in runs] for c in classes], runs)
    ax[1].set_title("AP50 per class: full test (compare with team)")
    dots(ax[2], [f"{c} (n={n_clean[c]}{', too few' if n_clean[c] < FEW else ''})" for c in classes],
         [[get(r, "clean", c) for r in runs] for c in classes], runs)
    ax[2].set_title("AP50 per class: clean test (honest)")
    legend(fig, runs)
    fig.suptitle("YOLO detector (stage 2: which condition, where)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS / "figures" / "yolo_overview.png")
    plt.close(fig)


def training_curves(runs):
    panels = [(["train/box_loss", "val/box_loss"], "Box loss (solid = train, dashed = clean val)"),
              (["train/cls_loss", "val/cls_loss"], "Class loss (solid = train, dashed = clean val)"),
              (["metrics/mAP50(B)", "metrics/mAP50-95(B)"], "Clean val mAP50 (solid) and mAP50-95 (dashed)")]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for a, (cols, name) in zip(ax, panels):
        for r in runs:
            h = r["hist"]
            for col, style in zip(cols, ("-", "--")):
                a.plot(h["epoch"], h[col], color=r["color"], linewidth=2, linestyle=style)
            a.axvline(r["best_epoch"], color=r["color"], linewidth=1, linestyle=":")
        a.set_title(name)
        a.set_xlabel("epoch (dotted line = best.pt)")
        tidy(a)
    legend(fig, runs)
    fig.suptitle("YOLO training curves", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS / "figures" / "yolo_training_curves.png")
    plt.close(fig)


def write_all():
    runs = load()
    if not runs:
        return
    classes = runs[0]["classes"]
    classes = list(classes.values()) if isinstance(classes, dict) else list(classes)
    n_clean = clean_counts(classes)
    (RESULTS / "figures").mkdir(exist_ok=True)
    overview(runs, classes, n_clean)
    training_curves(runs)

    w = 18
    out = []
    add = out.append

    def row(label, vals):
        add(f"{label:<44s}" + "".join(f"{v:>{w}s}" for v in vals))

    add("YOLO DETECTOR (STAGE 2): RESULTS")
    add("Aiden's YOLO on Daniel's oral_dataset. Runs on photos the stage-1 gate flags: which condition, and where.")
    add("Generated by models/yolo_caries/scripts/Aiden/report_yolo.py; rewritten after every run. Graphs: figures/yolo_*.png")
    add("=" * 100)
    add("")
    add("HOW TO READ THIS")
    add("- Daniel's full test is the number to compare with groupmates: everyone trains and tests on the same split.")
    add("- It contains copies of training photos (about 90% of its images with a condition, see find_leaks.py), so it")
    add("  overstates how well any model does on new photos. The clean test removes them; report both.")
    add(f"- The clean test is small for some classes ({', '.join(f'{c} {n}' for c, n in n_clean.items() if n < FEW) or 'none under 20'} images).")
    add("- best.pt is the epoch with the highest mAP50-95 on the clean val set (copies of training photos removed).")
    add("")
    row("", [r["model"] for r in runs])
    row("epochs run (best epoch)", [f"{r['settings']['epochs_run']} ({r['best_epoch']})" for r in runs])
    row("image size / batch", [f"{r['settings']['imgsz']} / {r['settings']['batch']}" for r in runs])
    for k, label in SETS:
        add("")
        add(label + (" (compare with team)" if k == "full" else ""))
        for m, ml in (("map50", "mAP50"), ("map50_95", "mAP50-95"), ("precision", "precision"), ("recall", "recall")):
            row(f"  {ml}", [f"{r['results'][k][m]:.3f}" for r in runs])
        for c in classes:
            row(f"    {c} AP50" + (f" (n={n_clean[c]})" if k == "clean" else ""),
                [f"{r['results'][k]['per_class'][c]['ap50']:.3f}" if c in r["results"][k]["per_class"] else "n/a" for r in runs])
    add("")
    add("FILES (models/yolo_caries/results/Aiden/)")
    for f, what in [("yolo_report.txt", "this report"), ("figures/yolo_overview.png", "mAP per test set, AP50 per class"),
                    ("figures/yolo_training_curves.png", "losses and clean-val mAP per epoch"),
                    ("<model>/weights/best.pt", "the trained detector (gitignored)"),
                    ("<model>/results.csv, *.png", "Ultralytics training log and plots"),
                    ("<model>/eval_<set>/", "Ultralytics PR curves and confusion matrices per test set"),
                    ("<model>/metrics.json, test_report.txt", "numbers in this report"),
                    ("_lists/", "clean val / test image lists and their data yaml files"),
                    ("../experiments.csv", "team log: one row per member + model (full-test numbers)")]:
        add(f"  {f:<40s}{what}")
    (RESULTS / "yolo_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_all()
