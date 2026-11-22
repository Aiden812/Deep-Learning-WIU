"""Show YOLO bounding boxes on their images, to check the boxes produced by qwen_clean_and_label.ipynb.

Default input is the notebook's YOLO dataset:
    data/processed/qwen_labels/Daniel/yolo/{images,labels}/   + classes.txt
    data/processed/qwen_labels/Daniel/labels.csv              (optional: source dataset, labels, model reason)
Any YOLO folder with images/ and labels/ subfolders works (--yolo-dir).

Each image is captioned with where it came from and what it was labelled as:
    line 1  source dataset(s) and their own label, e.g. "kaggle_oral: calculus" or "web: Caries + dentalai: Caries,
            Cavity" (an image can be in several datasets). Found through the notebook's file indexes in
            data/interim/Daniel/qwen_pipeline (--work-dir); the label comes from manifests/Daniel/classifier_split.csv
            (--split-csv), dentalai's annotation json, roboflow_oral / zenodo's YOLO files, the web-search folder, or
            the image's class folder.
    line 2  "Qwen: ..." - the image's final labels: from labels.csv when it lists the image, else from the
            notebook's label cache (so labels.csv is not needed), else the classes of its boxes.

Only images written by the current pipeline are shown. Until the notebook's "Write outputs" cell has run, the YOLO
folder still holds files from the earlier version, which were backed up to data/backups/qwen_pipeline_*. A label file
counts as new when it was written after the newest backup was made (--since to choose another time, --all to show
everything). Label caches older than label_v4 are then ignored too, so captions don't show the old answers.

Interactive viewer (default): a grid of images per page.
    right / space / n   next page          left / p    previous page
    home / end          first / last page
    type digits + Enter go to that page (Backspace edits, Esc cancels)
    click an image      open its review window:
                            verdict   Bad boxes (red frame) / Normal - no disease (green) / Discard image (grey)
                            issues    tick-boxes per class in the image (wrong positions, too big, not there) + general
                            details   free text, e.g. "Calculus was marked incorrectly as caries"; earlier answers
                                      are offered in its drop-down so they can be reused
                            Enter = save, Esc = cancel, "Clear review" = un-mark the image
    d                   show the model's reason under each image (toggle)
    q / Esc             quit
Reviews are saved to <yolo-dir>/../box_review.csv straight away (--flags to change), and reloaded next time,
so a review can be spread over several sessions. Columns: md5, image, dataset, rel, original (source datasets and
their labels), labels (Qwen), status (bad_boxes / normal / discard), issues ("; "-separated), note.

Usage:
    python models/classifier/scripts/Daniel/view_yolo_boxes.py
    python models/classifier/scripts/Daniel/view_yolo_boxes.py --classes Caries Mouth_Ulcer --shuffle
    python models/classifier/scripts/Daniel/view_yolo_boxes.py --datasets web dentalai --rows 2 --cols 3
    python models/classifier/scripts/Daniel/view_yolo_boxes.py --page 25          # start on page 25
    python models/classifier/scripts/Daniel/view_yolo_boxes.py --flagged          # re-check reviewed images only
    python models/classifier/scripts/Daniel/view_yolo_boxes.py --flagged discard  # ... only those marked discard
    python models/classifier/scripts/Daniel/view_yolo_boxes.py --save out_dir --limit 200   # write annotated JPEGs
    python models/classifier/scripts/Daniel/view_yolo_boxes.py --all             # also the old (backed-up) files
    python models/classifier/scripts/Daniel/view_yolo_boxes.py --since "2026-11-10 12:00"   # files written after this

Box-less (Normal / background) images are skipped unless --include-empty is given.
"""
import argparse
import csv
import datetime
import json
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO = Path(__file__).resolve().parents[4]
DEFAULT_YOLO = REPO / "data" / "processed" / "qwen_labels" / "Daniel" / "yolo"
DEFAULT_WORK = REPO / "data" / "interim" / "Daniel" / "qwen_pipeline"     # notebook's file indexes + label cache
DEFAULT_SPLIT = REPO / "manifests" / "Daniel" / "classifier_split.csv"   # original dataset labels
RAW = (REPO / "data" / "raw").resolve()
BACKUPS = REPO / "data" / "backups"                                     # data/backups/qwen_pipeline_<date>/
NEW_LABEL_VERSION = 4                                                   # label_v4 = first cache of the current prompts
DEFAULT_CLASSES = ["Calculus", "Caries", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]   # if classes.txt is missing
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128)]
Image.MAX_IMAGE_PIXELS = None


def read_boxes(label_file):
    """[(class_id, cx, cy, w, h), ...] from a YOLO .txt file.

    One box per line, coordinates normalised to 0-1 as the YOLO format prescribes. Deliberately
    lenient - a viewer must render whatever is on disk, including broken files: lines with fewer than
    5 fields or non-numeric junk are skipped silently (the notebook's writer is strict; this reader is
    the diagnostic). An empty list means an empty/missing label file, i.e. a Normal image.
    """
    boxes = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                boxes.append((int(float(parts[0])), *map(float, parts[1:5])))
            except ValueError:
                pass
    return boxes


# --- where each image came from -------------------------------------------------------------------------------------
def load_sources(work_dir):
    """md5 -> list of data/raw paths holding a copy of that image, from the notebook's file indexes.

    Two indexes are merged: index_raw.pkl (every image under data/raw, built by the notebook's
    indexing cell) and index_external.pkl (the scraped web copies). The image key everywhere is its
    content md5 - the same photo can sit in several datasets, and only a content hash can tie those
    copies together. Old notebook releases pickled a DataFrame instead of a record list, hence the
    to_dict fallback; an unreadable index is reported and skipped rather than failing the whole
    caption pass (captions then just show fewer sources).
    """
    out = {}
    for name in ["index_raw.pkl", "index_external.pkl"]:
        f = work_dir / name
        if not f.exists():
            continue
        try:
            with open(f, "rb") as fh:
                recs = pickle.load(fh)
            if hasattr(recs, "to_dict"):             # older notebook versions saved a DataFrame
                recs = recs.to_dict("records")
        except Exception as e:
            print(f"skipping {f} ({type(e).__name__}: {e})")
            continue
        for r in recs:
            if r.get("md5"):
                out.setdefault(r["md5"], []).append(Path(r["path"]))
    return out


def backup_time():
    """When the newest backup of the YOLO folder was made, or None if there is no backup.

    The notebook moves the old YOLO folder to data/backups/qwen_pipeline_<date>/ before writing a new
    one, so "newer than the newest backup" = "written by the current pipeline". The timestamp is the
    newest mtime inside the backup's labels folder - a plain copy, so every file carries the moment it
    was copied; the folder's own mtime would be just as good but file mtimes survive more copy tools.
    dirs[-1] is the newest backup because the folder names start with the date, so plain sorting is
    chronological. This becomes the default --since cutoff in main().
    """
    dirs = sorted(BACKUPS.glob("qwen_pipeline_*/processed/qwen_labels_Daniel/yolo/labels"))
    if not dirs:
        return None
    times = [p.stat().st_mtime for p in dirs[-1].iterdir()]
    return max(times) if times else None


def load_qwen_labels(work_dir, min_version=0):
    """md5 -> the model's label answer ({"labels": [...], "reason": ...}) from the notebook's label cache.

    The notebook appends one JSON line per answered request to label_v<N>__<prompt-hash>.jsonl caches;
    N is the prompt generation, so caches are read in version order (low to high) and later entries
    overwrite earlier ones - the newest prompt version wins for images that were re-labelled.
    Lines that failed (ok=False), lack a key, or have no "labels" field are skipped: the viewer shows
    answers, not error states. min_version comes from main(): when old files are hidden by --since,
    caches below NEW_LABEL_VERSION are also ignored so captions cannot mix prompt generations with
    the shown files.
    """
    out = {}
    caches = [(int(m.group(1)), f) for f in work_dir.glob("label_v*__*.jsonl") if (m := re.match(r"label_v(\d+)__", f.name))]
    for _, f in sorted(c for c in caches if c[0] >= min_version):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("key") and j.get("ok", True) and "labels" in j:
                    out[j["key"]] = j
    return out


def load_split_labels(path):
    """Repo-relative image path -> its label in the classifier split manifest (original dataset label).

    The split manifest carries a fine-grained `label` column and a coarser `binary` column
    (underscores become spaces for display). Used only by original_label() as the fallback for
    datasets that store labels nowhere else. A missing manifest is not an error: captions just lose
    that one source.
    """
    out = {}
    if path and path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if lab := r.get("label") or (r.get("binary") or "").replace("_", " "):
                    out[r["path"]] = lab
    return out


def rel_to_raw(path):
    """Path relative to data/raw, or None if the path is outside data/raw (e.g. inside the YOLO copy)."""
    try:
        return path.resolve().relative_to(RAW)
    except ValueError:
        return None


def raw_dataset(path):
    """Dataset folder under data/raw ("kaggle_oral", "external", ...), or the parent folder for other paths.

    The first path component under data/raw IS the dataset by convention - that is how setup.py
    unpacks each download and how the scraper stores classes.
    """
    rel = rel_to_raw(path)
    return rel.parts[0] if rel else path.parent.name


def original_label(path, split_labels):
    """What the image's own dataset says it shows (each dataset stores its label somewhere different).

    Resolution per dataset folder under data/raw: external/web images -> the class folder the scraper
    saved them into; dentalai -> the Supervisely annotation json's polygon class titles (minus the
    generic "Tooth" ring); mendeley_noncaries -> the dataset is caries-free by construction, its folders
    are just camera views; roboflow_oral/zenodo -> the sidecar YOLO .txt next to the ORIGINAL image
    (never the pipeline copy), classes mapped through data.yaml / "caries marked"; anything else ->
    the split-manifest label for this exact path if present, else the parent folder name.
    Every lookup degrades gracefully: a missing/corrupt sidecar falls through to the next source.
    """
    rel = rel_to_raw(path)
    if rel is None:
        return path.parent.name
    ds = rel.parts[0]
    if ds in ("external", "external_clean"):     # web images: the class folder they were scraped for
        return rel.parts[1] if len(rel.parts) > 2 else "?"
    if ds == "dentalai":                         # Supervisely json next to the image: classes of its polygons
        ann = path.parent.parent / "ann" / f"{path.name}.json"
        try:
            titles = {o.get("classTitle", "") for o in json.loads(ann.read_text(encoding="utf-8")).get("objects", [])}
            return ", ".join(sorted(titles - {"Tooth", ""})) or "no lesions marked"
        except (OSError, ValueError):
            pass
    if ds == "mendeley_noncaries":               # a caries-free dataset; its folders are camera views
        return f"non-caries ({rel.parent.name})"
    if ds in ("roboflow_oral", "zenodo"):        # YOLO files next to the images (classes from data.yaml for roboflow)
        lab = path.parent.parent / ("labels" if ds == "roboflow_oral" else "yolo") / f"{path.stem}.txt"
        if ds == "zenodo" and len(rel.parts) > 3 and rel.parts[1:3] == ("Dataset", "Images"):
            # zenodo's second copy: Dataset/Images/<group>/<view>/x.jpg -> Dataset/Annotations/Darknet_YOLO/<group>/<view>/x.txt
            lab = RAW / "zenodo" / "Dataset" / "Annotations" / "Darknet_YOLO" / Path(*rel.parts[3:]).with_suffix(".txt")
        if ds == "zenodo":
            return "caries marked" if lab.exists() and lab.read_text(encoding="utf-8").strip() else "no caries marked"
        yaml_file = path.parent.parent.parent / "data.yaml"
        m = re.search(r"^names:\s*\[(.*)\]", yaml_file.read_text(encoding="utf-8"), re.M) if yaml_file.exists() else None
        names = [n.strip().strip("'\"") for n in m.group(1).split(",")] if m else []
        try:
            ids = {int(float(l.split()[0])) for l in lab.read_text(encoding="utf-8").splitlines() if l.split()}
            return ", ".join(sorted(names[i] if i < len(names) else str(i) for i in ids)) or "no boxes"
        except (OSError, ValueError):
            pass
    if lab := split_labels.get(Path("data/raw", rel).as_posix()):
        return lab
    return rel.parent.name                       # kaggle_oral and others: class folder


def origin(item):
    """"kaggle_oral: calculus  +  web: Caries" for all copies of the image (computed on first use, then kept).

    One entry per dataset that holds a copy; the scraper's external folders are folded into one "web"
    group. Lazy because it touches annotation files on disk - most displayed items need it once per
    redraw, and the result is cached on the item dict (item["origin"]). "unknown source" when the
    image exists only inside the pipeline copy (no indexed raw source found).
    """
    if item.get("origin") is None:
        per_ds = {}
        for p in item["sources"]:
            ds = raw_dataset(p)
            name = "web" if ds in ("external", "external_clean") else ds
            per_ds.setdefault(name, set()).add(original_label(p, item["split"]).replace("_", " "))
        item["origin"] = "  +  ".join(f"{d}: {', '.join(sorted(l))}" for d, l in per_ds.items()) or "unknown source"
    return item["origin"]


def load_items(args):
    """Build one dict per image to show: boxes, source paths, origin/label captions, filtered by args.

    Filters applied here (in this order): --since (label file mtime - a label file older than the
    backup means the image was produced by the previous pipeline version and lives in the backup),
    box-less images dropped unless --include-empty, --classes (any box class overlap) and
    --datasets (any source-dataset overlap); --flagged/--limit are applied later in main() on the
    finished list.
    Merging per image (key = file stem = content md5): labels.csv contributes the meta row (first row
    per md5 wins) and may add a source path; external/external_clean are aliased with "web" so the
    --datasets filter matches however the scraper stored the copy.
    Caption fields: `labels` = labels.csv column (post-verification truth) else qwen cache else the
    classes actually drawn in the box file; `reason` = meta reason else cache reason.
    """
    images_dir, labels_dir = args.yolo_dir / "images", args.yolo_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        sys.exit(f"{args.yolo_dir} has no images/ and labels/ folders. Run qwen_clean_and_label.ipynb first, "
                 "or point --yolo-dir at a YOLO dataset.")
    classes_file = args.yolo_dir / "classes.txt"
    classes = classes_file.read_text(encoding="utf-8").split() if classes_file.exists() else DEFAULT_CLASSES

    meta, sources = {}, load_sources(args.work_dir)  # labels.csv: md5 -> first row; sources: md5 -> raw paths
    if args.labels_csv and args.labels_csv.exists():
        with open(args.labels_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta.setdefault(row["md5"], row)
                if row.get("rel") and (p := RAW / row["rel"]) not in sources.setdefault(row["md5"], []):
                    sources[row["md5"]].append(p)
    qwen = load_qwen_labels(args.work_dir, 0 if args.since is None else NEW_LABEL_VERSION)
    split = load_split_labels(args.split_csv)
    skipped_old = 0

    if args.classes and (bad := set(args.classes) - set(classes)):
        sys.exit(f"unknown class(es) {sorted(bad)}; classes are {classes}")
    want_cls = {classes.index(c) for c in args.classes} if args.classes else None
    items = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lf = labels_dir / f"{img.stem}.txt"
        if args.since is not None and (not lf.exists() or lf.stat().st_mtime <= args.since):
            skipped_old += 1                         # written by the earlier version (backed up): not shown
            continue
        boxes = read_boxes(lf) if lf.exists() else []
        if not boxes and not args.include_empty:
            continue
        if want_cls is not None and not want_cls & {b[0] for b in boxes}:
            continue
        m, q = meta.get(img.stem, {}), qwen.get(img.stem)
        srcs = sorted(set(sources.get(img.stem, [])))
        datasets = {raw_dataset(p) for p in srcs} | ({m["dataset"]} if m.get("dataset") else set())
        if datasets & {"external", "external_clean"}:
            datasets |= {"external", "external_clean", "web"}
        if args.datasets and not datasets & set(args.datasets):
            continue
        rel = m.get("rel") or next((r.as_posix() for p in srcs if (r := rel_to_raw(p))), "")
        items.append(dict(key=img.stem, image=img, boxes=boxes, sources=srcs, rel=rel, origin=None, split=split,
                          dataset=" + ".join(sorted({"web" if d.startswith("external") else d for d in datasets})),
                          labels=m["labels"].replace("|", ", ") if m.get("labels") else
                          ", ".join(q["labels"]) if q else
                          ", ".join(class_name(classes, c) for c in sorted({b[0] for b in boxes})),
                          reason=m.get("reason", "") if m else q.get("reason", "") if q else ""))
    if skipped_old and args.since is not None:
        print(f"{skipped_old} images from before {datetime.datetime.fromtimestamp(args.since):%Y-%m-%d %H:%M} hidden "
              "(earlier version, backed up; --all to show them)")
    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    return items, classes


def class_name(classes, c):
    """Class name for id c, tolerating ids outside classes.txt (shows "class 7" instead of crashing)."""
    return classes[c] if 0 <= c < len(classes) else f"class {c}"


def draw(item, classes, max_side=1024):
    """The image (EXIF-rotated, like the notebook) with its boxes and class names drawn on it.

    EXIF transpose matters: phone-camera JPEGs store pixels sideways + an orientation tag, and PIL
    ignores the tag, so without it the boxes would be rotated off the lesions. thumbnail() caps the
    render size (display only - originals can be 4000px); normalised cx/cy/w/h are scaled to the
    displayed size. Line width and font scale with the image so small thumbnails stay readable. The
    class label is painted on a filled colour plate above the box top (clamped to y>=0) so it is
    legible over any background.
    """
    im = ImageOps.exif_transpose(Image.open(item["image"])).convert("RGB")
    im.thumbnail((max_side, max_side))
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 300))
    try:
        font = ImageFont.truetype("arial.ttf", max(12, round(max(W, H) / 45)))
    except OSError:
        font = ImageFont.load_default()
    for c, cx, cy, bw, bh in item["boxes"]:
        col = COLORS[c % len(COLORS)]
        x1, y1, x2, y2 = (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        text = class_name(classes, c)
        tx, ty = x1, max(0, y1 - font.size - 2 * lw)
        l, t, r, b = d.textbbox((tx, ty), text, font=font)
        d.rectangle([l - 2, t - 2, r + 2, b + 2], fill=col)
        d.text((tx, ty), text, fill="white", font=font)
    return im


def caption(item):
    """Line 1: source dataset(s) and what they labelled the image as. Line 2: what Qwen labelled it as.

    Under the grid thumbnail; line 1 is truncated at 70 chars because origin strings can chain every
    dataset copy and the axes would overflow.
    """
    src = origin(item)
    src = src if len(src) <= 70 else src[:67] + "..."
    return f"{src}\nQwen: {item['labels'] or 'no label'}"


# --- reviews --------------------------------------------------------------------------------------------------------
FLAG_COLS = ["md5", "image", "dataset", "rel", "original", "labels", "status", "issues", "note"]
STATUSES = {                                     # status -> (label in the review window, frame colour)
    "bad_boxes": ("Bad boxes - needs fixing", "red"),
    "normal": ("Normal - no disease, the boxes should go", "limegreen"),
    "discard": ("Discard - remove the image from the dataset", "dimgray"),
}
CLASS_ISSUES = [                                 # (tick-box text, text saved in the issues column)
    ("wrong positions", "{c} boxes are in wrong positions"),
    ("too big / loose", "{c} boxes are too big / loose"),
    ("not there", "There is no {c}"),
]
GENERAL_ISSUES = ["Some lesions are not boxed", "Duplicate / overlapping boxes", "One box covers several lesions"]
ISSUE_SEP = "; "


def load_flags(path):
    """Load the review CSV into md5 -> row; rows without a verdict become "bad_boxes".

    Later rows overwrite earlier ones, so re-reviewing an image in a text editor after this tool
    still takes effect. Files written before the verdict columns existed only flagged bad boxes -
    the old rows are still accurate, just incomplete, so they get the implicit status instead of
    being dropped.
    """
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = {r["md5"]: r for r in csv.DictReader(f)}
    for r in rows.values():                      # files from before verdicts existed only held bad-box flags
        if r.get("status") not in STATUSES:
            r["status"] = "bad_boxes"
    return rows


def save_flags(path, flags):
    """Rewrite the whole review CSV from the in-memory dict (called after every single review).

    A full rewrite is intentional: the file is a few hundred small rows, and rewriting is simpler and
    safer than a read-modify-write merge - this tool is the only writer while it runs, so nothing can
    be lost, and a crash costs at most the one review in flight.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FLAG_COLS)
        w.writeheader()
        for r in flags.values():
            w.writerow({k: r.get(k) or "" for k in FLAG_COLS})


def review_dialog(parent, item, classes, current, note_history):
    """Modal Tk window to review one image. Returns a dict (status, issues, note), None to clear the review,
    or False if cancelled.

    Layout, top to bottom: provenance header (original sources, Qwen label, path, model reason);
    Verdict radios - the three STATUSES; Issues - one row of tick-boxes per class actually drawn in
    this image (CLASS_ISSUES with the class name substituted) plus the GENERAL_ISSUES rows, and any
    issue text already saved that no longer matches is kept as an extra box so old reviews don't lose
    information; free-text note combobox whose dropdown is every note ever written (most frequent
    first) so recurring observations are one click away.
    Behaviour: the issue boxes are enabled only while the verdict is "bad_boxes" (a "discard" verdict
    records no box complaints - sync()); the window pops up at the mouse cursor, clamped to stay
    fully on screen, and grab_set() makes it modal so grid keys can't fire behind it.
    Enter/Save returns the dict (issues joined only for bad_boxes); Esc, Cancel or the window ✕
    return False (nothing changes); "Clear review" (shown only when the image is already flagged)
    returns None, which main's caller uses to delete the row.
    """
    import tkinter as tk
    from tkinter import ttk

    result: dict = {"value": False}
    win = tk.Toplevel(parent)
    win.title("Review image")
    win.transient(parent.winfo_toplevel())
    win.resizable(False, False)

    ttk.Label(win, text=f"Original: {origin(item)}", font=("Segoe UI", 10, "bold"),
              wraplength=460).pack(anchor="w", padx=10, pady=(10, 0))
    ttk.Label(win, text=f"Qwen: {item['labels'] or 'no label'}", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10)
    ttk.Label(win, text=item["rel"] or item["key"], foreground="gray40", wraplength=460).pack(anchor="w", padx=10)
    if item["reason"]:
        ttk.Label(win, text=item["reason"], wraplength=460, foreground="gray25").pack(anchor="w", padx=10, pady=4)

    status = tk.StringVar(value=(current or {}).get("status", "bad_boxes"))
    vf = ttk.LabelFrame(win, text="Verdict")
    vf.pack(fill="x", padx=10, pady=4)
    for key, (label, _) in STATUSES.items():
        ttk.Radiobutton(vf, text=label, value=key, variable=status, command=lambda: sync()).pack(anchor="w", padx=8)

    chosen = set(filter(None, (current or {}).get("issues", "").split(ISSUE_SEP)))
    issue_vars, checks = {}, []

    def check(frame, text, saved, row, col):
        v = issue_vars.setdefault(saved, tk.BooleanVar(value=saved in chosen))
        cb = ttk.Checkbutton(frame, text=text, variable=v)
        cb.grid(row=row, column=col, sticky="w", padx=4)
        checks.append(cb)

    inf = ttk.LabelFrame(win, text="Issues (for bad boxes)")
    inf.pack(fill="x", padx=10, pady=4)
    in_image = sorted({b[0] for b in item["boxes"]})
    for r, c in enumerate(in_image):
        name = class_name(classes, c)
        ttk.Label(inf, text=name + ":", font=("Segoe UI", 9, "bold")).grid(row=r, column=0, sticky="w", padx=(8, 4))
        for col, (text, saved) in enumerate(CLASS_ISSUES, 1):
            check(inf, text, saved.format(c=name), r, col)
    row = len(in_image)
    extra = GENERAL_ISSUES + sorted(chosen - {s.format(c=class_name(classes, c)) for c in in_image
                                               for _, s in CLASS_ISSUES} - set(GENERAL_ISSUES))
    for i, text in enumerate(extra):
        check(inf, text, text, row + i // 2, 0 if i % 2 == 0 else 2)
        checks[-1].grid(columnspan=2, padx=(8, 4))

    nf = ttk.LabelFrame(win, text="Other / details (free text - earlier answers are in the drop-down)")
    nf.pack(fill="x", padx=10, pady=4)
    note = ttk.Combobox(nf, values=note_history, width=62)
    note.set((current or {}).get("note", ""))
    note.pack(fill="x", padx=8, pady=6)

    def sync():
        for cb in checks:
            cb.state(["!disabled"] if status.get() == "bad_boxes" else ["disabled"])

    def finish(value):
        result["value"] = value
        win.destroy()

    def save(_=None):
        issues = [s for s, v in issue_vars.items() if v.get()] if status.get() == "bad_boxes" else []
        finish(dict(status=status.get(), issues=ISSUE_SEP.join(issues), note=note.get().strip()))

    bf = ttk.Frame(win)
    bf.pack(fill="x", padx=10, pady=(4, 10))
    ttk.Button(bf, text="Save  (Enter)", command=save).pack(side="right")
    ttk.Button(bf, text="Cancel  (Esc)", command=lambda: finish(False)).pack(side="right", padx=6)
    if current:
        ttk.Button(bf, text="Clear review", command=lambda: finish(None)).pack(side="left")

    sync()
    win.bind("<Return>", save)
    win.bind("<Escape>", lambda _: finish(False))
    win.protocol("WM_DELETE_WINDOW", lambda: finish(False))
    win.update_idletasks()
    x = min(win.winfo_pointerx() + 10, win.winfo_screenwidth() - win.winfo_reqwidth() - 10)
    y = min(win.winfo_pointery() + 10, win.winfo_screenheight() - win.winfo_reqheight() - 60)
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    win.grab_set()
    note.focus_set()
    win.wait_window()
    return result["value"]


# --- modes ----------------------------------------------------------------------------------------------------------
def save_mode(items, classes, out_dir):
    """--save: write every image as an annotated JPEG named <dataset>__<md5>.jpg instead of opening a window.

    Same drawing as the grid, batched to disk - for eyeballing on a phone or sharing. Names stay
    unique because the md5 key is unique, and non-alphanumeric dataset names are flattened to
    underscores so the files survive every transfer tool. quality=90: these are inspection copies,
    not dataset inputs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items, 1):
        prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", it["dataset"]).strip("_")
        name = f"{prefix + '__' if prefix else ''}{it['key']}.jpg"
        draw(it, classes).save(out_dir / name, quality=90)
        if i % 50 == 0 or i == len(items):
            print(f"\r{i}/{len(items)} written", end="", flush=True)
    print(f"\n-> {out_dir}")


class Viewer:
    """Interactive grid viewer: one matplotlib page of images at a time, click an image to review it.

    Keyboard/mouse bindings are listed in the module docstring; reviews go straight to the flags CSV.
    Paging state is pure Python (page index + goto input), so every action just re-draws the figure;
    the flags dict is the single in-memory truth shared with review_dialog and re-persisted on change.
    """

    def __init__(self, items, classes, flags_path, rows, cols, start_page=1):
        """Open the window on start_page and block in plt.show() until the user quits.

        TkAgg is forced because review_dialog builds a plain Tk Toplevel - the figure window must be
        Tk for that window to share its interpreter and focus rules (the user's matplotlibrc could
        select any other backend). "backspace" is unbound from matplotlib's keymap.back so it can
        mean "edit page number" instead (on_key).
        The flags backfill loop repairs rows from earlier releases that predate the provenance
        columns: anything already reviewed (and present in this view) gets dataset/rel/original/labels
        filled from the matching item, so the enriched row lands in the CSV on the next save.
        """
        import matplotlib
        matplotlib.use("TkAgg")                      # the review window is a Tk window on top of the figure
        import matplotlib.pyplot as plt
        plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "backspace"]  # used for page input
        self.plt = plt
        self.items, self.classes, self.flags_path = items, classes, flags_path
        self.per_page = rows * cols
        self.pages = (len(items) + self.per_page - 1) // self.per_page
        self.page, self.details = min(max(start_page, 1), self.pages) - 1, False
        self.goto = ""                               # page number being typed
        self.flags = load_flags(flags_path)
        by_key = {it["key"]: it for it in items}
        for k, r in self.flags.items():              # fill in source info for reviews saved before it was recorded
            if (it := by_key.get(k)) and not r.get("original"):
                r.update(dataset=it["dataset"], rel=it["rel"], original=origin(it), labels=it["labels"])
        self.fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2), squeeze=False)
        self.axes = list(axes.flat)
        self.shown = []                              # item shown in each axis on the current page
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.show_page()
        plt.show()

    def frame(self, ax, color):
        """Coloured border around an axis (None = no border): shows the image's review verdict."""
        for s in ax.spines.values():
            s.set_visible(color is not None)
            s.set_edgecolor(color or "red")
            s.set_linewidth(5)

    def decorate(self, ax, it):
        """Title and coloured frame showing the image's review (if any).

        Title line(s) come from caption(); with the details toggle (d) the model's reason is appended
        (truncated to 90 chars so tiles stay uniform). A reviewed image additionally gets a
        "[STATUS - n note(s)]" header line where n counts ticked issues plus a free-text note, and its
        frame takes the status colour - the grid itself then doubles as the review dashboard.
        """
        title = caption(it)
        if self.details and it["reason"]:
            title += "\n" + it["reason"][:90]
        r = self.flags.get(it["key"])
        color = None
        if r:
            color = STATUSES[r["status"]][1]
            n = len([i for i in (r.get("issues") or "").split(ISSUE_SEP) if i]) + bool(r.get("note"))
            title = f"[{r['status'].replace('_', ' ').upper()}{f' - {n} note(s)' if n else ''}]\n{title}"
        ax.set_title(title, fontsize=8, wrap=True, color=color or "black")
        self.frame(ax, color)

    def show_page(self):
        """Redraw all axes with the current page's images (skipping unreadable ones gracefully)."""
        start = self.page * self.per_page
        self.shown = self.items[start:start + self.per_page]
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            self.frame(ax, None)
            ax.set_facecolor("white")
        for ax, it in zip(self.axes, self.shown):
            try:
                ax.imshow(draw(it, self.classes, max_side=800))
            except Exception as e:                   # unreadable image: say so instead of crashing
                ax.text(0.5, 0.5, f"cannot open\n{e}"[:80], ha="center", va="center", transform=ax.transAxes)
            self.decorate(ax, it)
        for ax in self.axes[len(self.shown):]:
            ax.axis("off")
        self.update_title()
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))
        self.fig.canvas.draw_idle()

    def update_title(self):
        """Window title: page position, review counts, and the key hints (or the goto input)."""
        if self.goto:
            hint = f"   go to page: {self.goto}_   (Enter = go, Esc = cancel)"
        else:
            hint = "\n(click = review image, ←/→ = page, type number + Enter = go to page, d = reasons, q = quit)"
        counts = {s: 0 for s in STATUSES}
        for r in self.flags.values():
            counts[r["status"]] += 1
        reviewed = ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in counts.items())
        self.fig.suptitle(f"page {self.page + 1}/{self.pages}  |  {len(self.items)} images  |  reviewed: {reviewed}"
                          + hint, fontsize=10)

    def on_key(self, e):
        """Any key press: digits/backspace/enter/esc drive the "go to page" input, everything else is paging.

        While digits are being typed the window title echoes them (update_title shows the goto input);
        Enter jumps to that page (clamped), Esc abandons the input, and ANY other key drops the typed
        digits and falls through to normal navigation - so a stray digit never blocks the arrow keys.
        """
        if e.key is not None and e.key.isdigit():   # build up a page number to jump to
            self.goto += e.key
        elif self.goto and e.key == "backspace":
            self.goto = self.goto[:-1]
        elif self.goto and e.key == "escape":
            self.goto = ""
        elif self.goto and e.key == "enter":
            self.page = min(max(int(self.goto), 1), self.pages) - 1
            self.goto = ""
            self.show_page()
            return
        else:
            self.goto = ""
            self.on_nav_key(e)
            return
        self.update_title()
        self.fig.canvas.draw_idle()

    def on_nav_key(self, e):
        """Paging / details / quit keys (only called when no goto input is being typed)."""
        if e.key in ("right", " ", "n", "pagedown"):
            self.page = min(self.page + 1, self.pages - 1)
        elif e.key in ("left", "p", "pageup"):
            self.page = max(self.page - 1, 0)
        elif e.key == "home":
            self.page = 0
        elif e.key == "end":
            self.page = self.pages - 1
        elif e.key == "d":
            self.details = not self.details
        elif e.key in ("q", "escape"):
            self.plt.close(self.fig)
            return
        else:
            return
        self.show_page()

    def on_click(self, e):
        """Left click on an image: open the review dialog and save/clear its verdict in the flags CSV.

        Guards: left button only, only on a filled axis, and not while the matplotlib toolbar is in
        zoom/pan mode (toolbar.mode) - otherwise dragging to zoom would open dialogs by accident.
        The dialog's note dropdown is every note in the flags file ordered by how often it was used.
        After the dialog the Tk canvas regains focus, or key bindings would keep going to the dead
        dialog. False (cancel) changes nothing; None deletes the row; a verdict dict is stored whole
        (provenance included, so the CSV is self-describing) and immediately written to disk. Only
        the clicked tile is re-decorated - no page reload, so paging state survives every review.
        """
        toolbar = self.fig.canvas.toolbar
        if e.button != 1 or e.inaxes not in self.axes or (toolbar is not None and toolbar.mode):
            return                                   # left clicks only, and not while zooming / panning
        i = self.axes.index(e.inaxes)
        if i >= len(self.shown):
            return
        it = self.shown[i]
        notes = Counter(r["note"] for r in self.flags.values() if r.get("note"))
        tk_widget = self.fig.canvas.get_tk_widget()
        res = review_dialog(tk_widget, it, self.classes, self.flags.get(it["key"]),
                            [n for n, _ in notes.most_common()])
        tk_widget.focus_set()                        # so the figure gets key presses again
        if res is False:
            return
        if res is None:
            self.flags.pop(it["key"], None)
        else:
            self.flags[it["key"]] = dict(md5=it["key"], image=str(it["image"]), dataset=it["dataset"], rel=it["rel"],
                                         original=origin(it), labels=it["labels"], status=res["status"],
                                         issues=res["issues"],
                                         note=res["note"])
        save_flags(self.flags_path, self.flags)
        self.decorate(e.inaxes, it)
        self.update_title()
        self.fig.canvas.draw_idle()


def main():
    """Resolve --since, collect the items, then run the save mode or open the interactive Viewer.

    --since resolution: --all shows everything (since=None); an explicit "YYYY-MM-DD HH:MM" is parsed
    to a timestamp; otherwise the newest qwen_pipeline backup's time is used - the point in time the
    current label-generation pipeline last ran, so the default view is "what the current version
    wrote". Defaults hang off --yolo-dir (labels.csv and box_review.csv live one level above the
    YOLO images/labels folders).
    Post-filter order: --flagged (any recorded status, or only the listed ones) narrows to reviewed
    images, then --limit truncates - so "--flagged --limit 40" reviews the first 40 flagged images.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO, help="folder with images/ and labels/")
    ap.add_argument("--labels-csv", type=Path, default=None,
                    help="labels.csv with source dataset and model reason (default: <yolo-dir>/../labels.csv)")
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK,
                    help="notebook work folder with index_*.pkl and the label cache (to find source datasets)")
    ap.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT,
                    help="classifier split manifest with each source image's original label")
    ap.add_argument("--classes", nargs="+", help="only images with a box of one of these classes")
    ap.add_argument("--datasets", nargs="+", help="only images from these source datasets (needs labels.csv)")
    ap.add_argument("--include-empty", action="store_true", help="also show images without boxes (Normal backgrounds)")
    ap.add_argument("--flagged", nargs="*", choices=list(STATUSES), metavar="STATUS",
                    help="only images already reviewed; optionally only these statuses (bad_boxes, normal, discard)")
    ap.add_argument("--flags", type=Path, default=None, help="review file (default: <yolo-dir>/../box_review.csv)")
    ap.add_argument("--shuffle", action="store_true", help="random order instead of sorted by file name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="show at most this many images")
    ap.add_argument("--page", type=int, default=1, help="page to open on (1-based)")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--save", type=Path, metavar="DIR", help="write annotated JPEGs to DIR instead of opening a window")
    ap.add_argument("--all", action="store_true", help="also show files written before the newest backup (old version)")
    ap.add_argument("--since", metavar="'YYYY-MM-DD HH:MM'",
                    help="only show label files written after this time (default: when the newest backup was made)")
    args = ap.parse_args()

    if args.all:
        args.since = None
    elif args.since:
        args.since = datetime.datetime.fromisoformat(args.since).timestamp()
    else:
        args.since = backup_time()
        if args.since is None:
            print(f"no backup in {BACKUPS}: showing every image")

    args.labels_csv = args.labels_csv or args.yolo_dir.parent / "labels.csv"
    args.flags = args.flags or args.yolo_dir.parent / "box_review.csv"
    items, classes = load_items(args)
    if args.flagged is not None:
        want = set(args.flagged or STATUSES)
        flagged = {k for k, r in load_flags(args.flags).items() if r["status"] in want}
        items = [it for it in items if it["key"] in flagged]
    if args.limit:
        items = items[:args.limit]
    if not items:
        sys.exit("No images match (box-less images are skipped unless --include-empty).")
    print(f"{len(items)} images | classes: {', '.join(classes)}")

    if args.save:
        save_mode(items, classes, args.save)
    else:
        Viewer(items, classes, args.flags, args.rows, args.cols, args.page)


if __name__ == "__main__":
    main()
