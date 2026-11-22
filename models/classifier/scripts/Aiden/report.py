"""Comparison report and graphs for the stage-1 gate (healthy vs has an issue): DenseNet-121 vs DINOv2.

    python models/classifier/scripts/Aiden/report.py      # rebuild from whatever models have been trained

Reads results/Aiden/<model>/metrics.json and *_predictions.csv (written by train.py), plus Daniel's gate results for the
cross-source comparison, and
rewrites results/Aiden/comparison_report.txt and results/Aiden/figures/comparison_*.png. train.py calls write_all()
after every training run, so the report and graphs always match the latest models.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results" / "Aiden"
DANIEL = RESULTS.parent / "Daniel" / "binary" / "test_metrics.json"   # his gate, scored on the same xsrc_test
# Categorical slots 1-2 of the default palette, validated with the dataviz skill's checker (CVD dE 24.7, contrast >= 3:1)
COLORS = {"densenet121": "#2a78d6", "dinov2_vitb14": "#eb6834"}
EXTRA = ["#1baf7a", "#eda100"]   # slots 3-4, only if more models are trained
INK, MUTED, GRID = "#1a1a19", "#5f5e5a", "#e4e3df"
FIGURES = ["comparison_overview.png", "comparison_curves.png", "comparison_confusion.png", "comparison_training_curves.png"]
FEW = 20   # conditions with fewer clean test images than this are flagged as unreliable
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.dpi": 150, "legend.frameon": False})


def load(root):
    runs = []
    for m in sorted(root.glob("*/metrics.json")):
        r = json.loads(m.read_text())
        if r.get("task") != "binary gate: has_issue" or "xsrc_test" not in r:
            continue   # an older run (multi-label, or before the cross-source sets): not comparable
        r["color"] = COLORS.get(r["arch"]) or EXTRA[len(runs) % len(EXTRA)]
        for name in ("val", "test"):
            f = m.parent / f"{name}_predictions.csv"
            r[f"{name}_pred"] = pd.read_csv(f) if f.exists() else None
        runs.append(r)
    return runs


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend(fig, runs):
    handles = [plt.Line2D([], [], color=r["color"], marker="o", linewidth=2, markersize=7) for r in runs]
    fig.legend(handles, [r["name"] for r in runs], loc="upper right", ncol=len(runs), fontsize=10)


def title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold")


def dots(ax, labels, values, runs, xlim, fmt="{:.3f}"):
    """Dot plot: one row per label, one dot per model, joined by a grey line. Truthful on a zoomed axis, unlike bars."""
    y = np.arange(len(labels))[::-1]
    v = np.array(values, dtype=float)   # rows x models
    off = 0.12 * (np.arange(len(runs)) - (len(runs) - 1) / 2)   # small vertical nudge so equal scores don't hide each other
    for i in range(len(labels)):
        ax.plot(v[i], y[i] - off, color=GRID, linewidth=3, zorder=1, solid_capstyle="round")
    for j, r in enumerate(runs):
        ax.scatter(v[:, j], y - off[j], s=60, color=r["color"], edgecolor="white", linewidth=1.5, zorder=3)
    for i in range(len(labels)):   # label each row's best value only
        j = int(np.nanargmax(v[i]))
        ax.annotate(fmt.format(v[i, j]), (v[i, j], y[i]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    tidy(ax, grid="x")


def bars(ax, labels, values, runs, fmt="{:.0%}", xmax=1.15):
    """Grouped horizontal bars from zero (for rates, where zero is meaningful)."""
    y = np.arange(len(labels))[::-1]
    h = 0.8 / len(runs)
    for j, r in enumerate(runs):
        pos = y + (len(runs) - 1) / 2 * h - j * h
        ax.barh(pos, [v[j] for v in values], height=h - 0.04, color=r["color"], edgecolor="white", linewidth=1)
        for p, v in zip(pos, values):
            ax.annotate(fmt.format(v[j]), (v[j], p), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    tidy(ax, grid="x")


def overview(root, runs):
    r0 = runs[0]
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    rows = [("AUROC", "auroc"), ("Average precision", "ap"), ("Sensitivity (issues caught)", "sensitivity"),
            ("Specificity (healthy cleared)", "specificity"), ("Precision (flags right)", "precision"), ("Accuracy", "accuracy")]
    dots(ax[0], [n for n, _ in rows], [[r[k] for r in runs] for _, k in rows], runs, (0.0, 1.02))
    ax[0].set_title(f"Clean test ({r0['n_test']} images), higher is better")
    rows = [("Clean test", None), ("Cross-source test", "xsrc_test"), ("Person-labelled only", "person"),
            ("Full test (leaky)", "full_test"), ("Clean val", "val"), ("Cross-source val", "xsrc_val")]
    v = [[(r[k]["auroc"] if k else r["auroc"]) for r in runs] for _, k in rows]
    dots(ax[1], [n for n, _ in rows], v, runs, (min(map(min, v)) - 0.01, 1.002), fmt="{:.4f}")
    ax[1].set_title("AUROC on each evaluation set")
    conds = list(r0["passed_to_yolo"])
    labels = [f"{c} (n={r0['passed_to_yolo'][c]['n']}{', too few' if r0['passed_to_yolo'][c]['n'] < FEW else ''})" for c in conds]
    bars(ax[2], labels + ["Healthy (wrongly passed)"],
         [[r["passed_to_yolo"][c]["share"] for r in runs] for c in conds] + [[1 - r["specificity"] for r in runs]], runs)
    ax[2].set_title("Share passed on to YOLO (clean test)")
    legend(fig, runs)
    title(fig, "Stage-1 gate (healthy vs has an issue): model comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(root / "figures" / "comparison_overview.png")
    plt.close(fig)


def curves(root, runs):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    y = runs[0]["test_pred"].has_issue.values == 1
    ax[0].plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
    ax[1].axhline(y.mean(), color=MUTED, linestyle=":", linewidth=1)
    ax[1].annotate("chance", (0.0, y.mean()), xytext=(2, 3), textcoords="offset points", fontsize=7, color=MUTED)
    ax[2].plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
    ax[2].annotate("perfect calibration", (0.6, 0.64), rotation=37, fontsize=7, color=MUTED)
    edges = np.linspace(0, 1, 11)
    for r in runs:
        p = r["test_pred"].p_issue.values
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=r["color"], linewidth=2, label=f"{r['name']}  AUROC {r['auroc']:.4f}")
        ax[0].scatter([1 - r["specificity"]], [r["sensitivity"]], s=70, color=r["color"], edgecolor="white", linewidth=1.5, zorder=3)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=r["color"], linewidth=2, drawstyle="steps-post", label=f"{r['name']}  AP {r['ap']:.4f}")
        ax[1].scatter([r["sensitivity"]], [r["precision"]], s=70, color=r["color"], edgecolor="white", linewidth=1.5, zorder=3)
        b = np.clip(np.digitize(p, edges) - 1, 0, 9)
        keep = [k for k in range(10) if (b == k).sum() >= 10]   # skip bins with too few predictions to read
        ax[2].plot([p[b == k].mean() for k in keep], [y[b == k].mean() for k in keep], color=r["color"], linewidth=2,
                   marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    ax[0].set(xlabel="false alarm rate (1 - specificity)", ylabel="sensitivity", xlim=(0, 1), ylim=(0, 1.02))
    ax[0].set_title("ROC (dot = saved threshold)")
    ax[1].set(xlabel="recall (issues caught)", ylabel="precision (flags right)", xlim=(0, 1.02), ylim=(0, 1.02))
    ax[1].set_title("Precision-recall (dot = saved threshold)")
    ax[2].set(xlabel="predicted probability of an issue", ylabel="observed share with an issue", xlim=(0, 1), ylim=(0, 1))
    ax[2].set_title("Calibration")
    for a in ax[:2]:
        a.legend(loc="lower right" if a is ax[0] else "lower left", fontsize=8)
    for a in ax:
        tidy(a, grid="both")
    title(fig, f"Gate curves, clean test ({runs[0]['n_issue']} with an issue, {runs[0]['n_healthy']} healthy)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(root / "figures" / "comparison_curves.png")
    plt.close(fig)


def confusion(root, runs):
    fig, ax = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6))
    ax = np.atleast_1d(ax)
    for a, r in zip(ax, runs):
        c = r["confusion"]
        cells = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        share = cells / cells.sum(1, keepdims=True)   # shade by row share: each true class is read on its own
        a.imshow(share, cmap="Greys", vmin=0, vmax=1.6)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cells[i, j]}\n{share[i, j]:.1%} of row", ha="center", va="center", fontsize=10, color=INK)
        a.set_xticks([0, 1], ["healthy", "issue"])
        a.set_yticks([0, 1], ["healthy", "issue"])
        a.set_xlabel("gate says")
        a.set_ylabel("truth")
        a.set_title(f"{r['name']} (threshold {r['threshold']:.2f})", color=r["color"])
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
    title(fig, "Confusion matrices, clean test")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(root / "figures" / "comparison_confusion.png")
    plt.close(fig)


def training_curves(root, runs):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for a, (key, name) in zip(ax, [("train_loss", "Training loss"), ("heldout_loss", "Clean val loss"), ("heldout_auroc", "Clean val AUROC")]):
        for r in runs:
            h = np.array([[e[key] for e in run] for run in r["history"]])
            ep = np.arange(1, h.shape[1] + 1)
            a.plot(ep, h.mean(0), color=r["color"], linewidth=2, marker="o", markersize=5, markeredgecolor="white")
        a.set_title(name)
        a.set_xlabel("epoch")
        tidy(a)
    legend(fig, runs)
    title(fig, "Training curves (val scored after every epoch; it never steers training)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(root / "figures" / "comparison_training_curves.png")
    plt.close(fig)


def daniel():
    """Daniel's gate on xsrc_test at his threshold, or None if his results aren't in the repo."""
    if not DANIEL.exists():
        return None
    m = json.loads(DANIEL.read_text())
    return dict(m["test_at_threshold"]["xsrc_test"], backbone=m["backbone"], target=m["target_sensitivity"])


def write_all(root=RESULTS):
    runs = load(root)
    if not runs:
        return
    (root / "figures").mkdir(exist_ok=True)
    for f in (root / "figures").glob("comparison_*.png"):   # drop graphs from an older report layout
        if f.name not in FIGURES:
            f.unlink()
    overview(root, runs)
    if all(r["test_pred"] is not None for r in runs):
        curves(root, runs)
    confusion(root, runs)
    training_curves(root, runs)
    write_comparison(root, runs)


def write_comparison(root, runs):
    """The single text report comparing every trained model."""
    w = 22
    out = []
    add = out.append
    r0 = runs[0]

    def row(label, vals):
        add(f"{label:<42s}" + "".join(f"{v:>{w}s}" for v in vals))

    def best(key, higher=True):
        vals = [round(r[key], 4) for r in runs]
        top = (max if higher else min)(vals)
        return "tie" if vals.count(top) > 1 else runs[vals.index(top)]["name"]

    add("STAGE-1 GATE (HEALTHY vs HAS AN ISSUE): MODEL COMPARISON")
    add("Aiden's binary classifier (IT2226). Photos it flags go on to the YOLO detector, which finds which condition and where.")
    add("Generated by models/classifier/scripts/Aiden/report.py; rewritten every time a model is retrained. Do not edit by hand.")
    add("Graphs: figures/comparison_*.png")
    add("=" * 100)
    add("")
    add("1. SUMMARY")
    add("-" * 100)
    if len(runs) > 1:
        a_, b_ = sorted(runs, key=lambda r: -r["auroc"])[:2]
        overlap = a_["auroc_ci"][0] <= b_["auroc_ci"][1] and b_["auroc_ci"][0] <= a_["auroc_ci"][1]
        add(f"AUROC on the clean test: {a_['name']} {a_['auroc']:.4f}, {b_['name']} {b_['auroc']:.4f}. The 95% intervals "
            + ("overlap, so this gap alone is not conclusive." if overlap else "do not overlap."))
        add(f"At thresholds set to catch {r0['settings']['target_sensitivity']:.0%} of issues in clean val and cross-source val: {a_['name']} catches "
            f"{a_['sensitivity']:.1%} of test issues and clears {a_['specificity']:.1%} of healthy mouths; {b_['name']} "
            f"{b_['sensitivity']:.1%} and {b_['specificity']:.1%}.")
        missed = {r["name"]: r["confusion"]["fn"] for r in runs}
        add("Issues missed by the gate (never reach YOLO): " + ", ".join(f"{k} {v}" for k, v in missed.items()) + f" of {r0['n_issue']}.")
    else:
        add(f"Only {r0['name']} has been trained so far.")
    x = [f"{r['name']} {r['xsrc_test']['auroc']:.4f}" for r in runs]
    d = daniel()
    add(f"Cross-source test (non-Roboflow photos, both classes from the same sources): AUROC {', '.join(x)}"
        + (f"; Daniel's gate ({d['backbone']}) {d['auroc']:.4f} on the same images." if d else "."))
    add("")
    add("2. METHOD (identical for every model; only the backbone and its learning rate differ)")
    add("-" * 100)
    for line in [
        "Pipeline  Stage 1 (this model): has an issue, or healthy? Flagged photos go to stage 2, YOLO (which condition, where).",
        "Data      Daniel's oral_dataset (data/raw/oral_dataset), target has_issue. Sources: roboflow_oral, zenodo, dentalai,",
        "          kaggle_oral, mendeley_noncaries, screened web images; labelled by people or Qwen (labels.csv: label_by).",
        f"Split     Daniel's binary split (manifests/Daniel/binary_classifier_split.csv). Train: {r0['n_train']} images (his train",
        "          minus two cross-source sets carved from non-Roboflow photo groups: xsrc_val, xsrc_test). Val/test: only images",
        "          with no copy of a training photo (find_leaks.py: SIFT keypoint matching on roboflow_oral). About 90% of the",
        f"          val/test images with an issue were copies. Clean val: {r0['n_val']} images. Clean test: {r0['n_test']} of "
        f"{r0['n_test_full']}. Both test sets used once at the end.",
        "Training  One sigmoid output. Fixed number of epochs (no epoch picked on validation), AdamW wd 0.05, one-cycle cosine,",
        "          mixed precision, BCE with pos_weight + label smoothing 0.05, each photo group equally likely per epoch.",
        "          Augmentation: random resized crop 0.5-1.0, flip, rotation 20, brightness/contrast 0.2 (no hue: discolouration is a colour).",
        f"Decision  Probability calibrated on clean val + cross-source val. Threshold: the highest that still catches "
        f"{r0['settings']['target_sensitivity']:.0%} of issues in each,",
        "          because a missed issue never reaches YOLO while a false alarm only costs one YOLO run. Test predictions",
        "          average the image and its mirror.",
    ]:
        add(line)
    add("")
    row("Settings", [r["name"] for r in runs])
    row("  image size", [str(r["settings"]["img"]) for r in runs])
    row("  epochs", [str(r["settings"]["epochs"]) for r in runs])
    row("  backbone learning rate", [f"{r['settings']['lr']:g}" for r in runs])
    row("  threshold", [f"{r['threshold']:.3f}" for r in runs])
    add("")
    add(f"3. RESULTS, CLEAN TEST ({r0['n_test']} images: {r0['n_issue']} with an issue, {r0['n_healthy']} healthy)")
    add("   (graphs: figures/comparison_overview.png, comparison_curves.png, comparison_confusion.png)")
    add("-" * 100)
    row("", [r["name"] for r in runs])
    row("AUROC", [f"{r['auroc']:.4f}" for r in runs])
    row("  95% CI", [f"{r['auroc_ci'][0]:.4f}-{r['auroc_ci'][1]:.4f}" for r in runs])
    row("Average precision", [f"{r['ap']:.4f}" for r in runs])
    row("Sensitivity (issues caught)", [f"{r['sensitivity']:.1%}" for r in runs])
    row("  95% CI", [f"{r['sensitivity_ci'][0]:.1%}-{r['sensitivity_ci'][1]:.1%}" for r in runs])
    row("Specificity (healthy cleared)", [f"{r['specificity']:.1%}" for r in runs])
    row("Precision (flags that are real issues)", [f"{r['precision']:.1%}" for r in runs])
    row("Accuracy", [f"{r['accuracy']:.1%}" for r in runs])
    row("Confusion: issues caught / missed", [f"{r['confusion']['tp']} / {r['confusion']['fn']}" for r in runs])
    row("Confusion: healthy cleared / false alarms", [f"{r['confusion']['tn']} / {r['confusion']['fp']}" for r in runs])
    if len(runs) > 1:
        add(f"Better: AUROC {best('auroc')}; sensitivity {best('sensitivity')}; specificity {best('specificity')}.")
    add("")
    add("AUROC on the other evaluation sets:")
    row("  person-labelled clean test only", [f"{r['person']['auroc']:.4f}" for r in runs])
    row(f"  Daniel's full test (leaky, {r0['n_test_full']})", [f"{r['full_test']['auroc']:.4f}" for r in runs])
    row(f"  clean val ({r0['n_val']}, used for the threshold)", [f"{r['val']['auroc']:.4f}" for r in runs])
    add("")
    xs = r0["xsrc_test"]["confusion"]
    add(f"3b. RESULTS, CROSS-SOURCE TEST ({sum(xs.values())} non-Roboflow images: {xs['tp'] + xs['fn']} with an issue, "
        f"{xs['tn'] + xs['fp']} healthy)")
    add("-" * 100)
    add("The clean test has every issue photo from Roboflow and every healthy one from elsewhere, so it can't tell disease")
    add("from photo style. This set has both classes from the same non-Roboflow sources (Daniel's xsrc_test), and Daniel's")
    add("gate is scored on the same images, so this is the fairest comparison with his model.")
    d = daniel()
    row("", [r["name"] for r in runs] + (["Daniel's gate"] if d else []))
    for label, k, f in [("AUROC", "auroc", "{:.4f}"), ("Average precision", "ap", "{:.4f}"),
                        ("Sensitivity (issues caught)", "sensitivity", "{:.1%}"), ("Specificity (healthy cleared)", "specificity", "{:.1%}")]:
        row(label, [f.format(r["xsrc_test"][k]) for r in runs] + ([f.format(d[k])] if d else []))
    row("  AUROC 95% CI", [f"{r['xsrc_test']['auroc_ci'][0]:.4f}-{r['xsrc_test']['auroc_ci'][1]:.4f}" for r in runs] + (["-"] if d else []))
    add("AUROC within one source (both classes are the same kind of photo; the cleanest 'disease, not source' check):")
    for src, v in r0["xsrc_per_source"].items():
        row(f"  {src} ({v['n_issue']} issue, {v['n_healthy']} healthy)",
            [("n/a" if r["xsrc_per_source"][src]["auroc"] is None else f"{r['xsrc_per_source'][src]['auroc']:.4f}") for r in runs])
    if d:
        add(f"Daniel's numbers are at his own threshold (also set to catch {d['target']:.0%} of issues; backbone {d['backbone']}), from {DANIEL.relative_to(RESULTS.parents[3]).as_posix()}.")
    add("")
    add("4. WHAT REACHES YOLO, PER CONDITION (clean test)")
    add("-" * 100)
    add("Share of images with each condition that the gate flags. A miss here is a miss for the whole pipeline.")
    for c, v in r0["passed_to_yolo"].items():
        row(f"  {c} ({v['n']} images{'; too few to trust' if v['n'] < FEW else ''})", [f"{r['passed_to_yolo'][c]['share']:.1%}" for r in runs])
    row("  healthy, wrongly passed", [f"{1 - r['specificity']:.1%}" for r in runs])
    add("")
    add("5. LIMITATIONS")
    add("-" * 100)
    few = [f"{c} {v['n']}" for c, v in r0["passed_to_yolo"].items() if v["n"] < FEW]
    for line in [
        "- In the clean test every issue photo is from Roboflow, so it mostly measures 'Roboflow photo or not'; read it",
        "  together with the cross-source test (3b). mendeley_noncaries only has healthy photos, even there.",
        f"- The clean test is small for some conditions ({', '.join(few) if few else 'none under 20'} images), "
        "because copies of training photos fill Daniel's val/test.",
        "- Most healthy val/test photos were judged healthy by Qwen, not a person, so a few may hide a small finding.",
        "- Each model was trained once with one learning rate; no other settings were tuned for this comparison.",
        "- Confidence intervals resample images and ignore that some are related, so they are somewhat optimistic.",
        "- The training photos are not from Singapore patients.",
    ]:
        add(line)
    add("")
    add("6. FILES (models/classifier/results/Aiden/)")
    add("-" * 100)
    for f, what in [
        ("comparison_report.txt", "this report"),
        ("figures/comparison_overview.png", "scores, AUROC per evaluation set, share passed to YOLO per condition"),
        ("figures/comparison_curves.png", "ROC, precision-recall and calibration"),
        ("figures/comparison_confusion.png", "confusion matrices at the saved thresholds"),
        ("figures/comparison_training_curves.png", "loss and clean-val AUROC per epoch"),
        ("<model>/model.pt", "the trained gate (gitignored)"),
        ("<model>/metrics.json", "numbers in this report"),
        ("<model>/test_report.txt", "full test report for that model"),
        ("<model>/train_log.txt", "training log"),
        ("<model>/val_, test_, xsrc_test_predictions.csv", "calibrated probabilities on clean val / clean test / cross-source test"),
        ("../../../data/processed/oral_dataset_leaks.csv", "which val/test images are copies of training photos"),
    ]:
        add(f"  {f:<52s}{what}")
    (root / "comparison_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_all()
