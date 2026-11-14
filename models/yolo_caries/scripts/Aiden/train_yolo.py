"""Stage 2 (Aiden): YOLO detector on Daniel's oral_dataset: which condition, and where.

    python models/classifier/scripts/Aiden/find_leaks.py               # once (shared with the classifier)
    python models/yolo_caries/scripts/Aiden/train_yolo.py               # trains yolo11n (as in Daniel's README), then evaluates
    python models/yolo_caries/scripts/Aiden/train_yolo.py --model yolo11s.pt --epochs 60
    python models/yolo_caries/scripts/Aiden/train_yolo.py --eval-only   # re-evaluate the saved best.pt

Train: Daniel's yolo_train.txt (healthy mouths capped at 10%), unchanged, so the model is comparable with the team's.
Val (picks best.pt): Daniel's val minus copies of training photos (find_leaks.py), so the saved epoch is not chosen by
memorised copies. Test, reported three ways:
  full    Daniel's data.yaml test split: the number to compare with groupmates (inflated by copies of training photos)
  person  Daniel's data_person.yaml test split (images labelled entirely by people), as his README asks
  clean   test images with no copy in train: the honest score

Output, overwritten by every run: models/yolo_caries/results/Aiden/<model>/ (Ultralytics run: weights/best.pt,
results.csv, its plots; plus metrics.json and test_report.txt), and report_yolo.py rewrites yolo_report.txt + graphs.
The run is also logged in the team's models/yolo_caries/results/experiments.csv (one row per member + model, replaced).
"""
import argparse
import ctypes
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "data/raw/oral_dataset"
LEAKS = REPO / "data/processed/oral_dataset_leaks.csv"
RESULTS = REPO / "models/yolo_caries/results/Aiden"
TEAM_LOG = REPO / "models/yolo_caries/results/experiments.csv"
SEED = 42


def write_lists(work):
    """Clean val / test image lists (absolute paths) and a data yaml per evaluation set."""
    d = pd.read_csv(DATA / "labels.csv", keep_default_na=False).merge(pd.read_csv(LEAKS)[["md5", "leaked"]], on="md5", how="left")
    d = d[d.in_yolo == 1]
    d["leaked"] = d.leaked.fillna(False).astype(bool)
    names = yaml.safe_load((DATA / "data.yaml").read_text())["names"]
    lists = {"val_clean": (d.split == "val") & ~d.leaked, "test_clean": (d.split == "test") & ~d.leaked,
             "test_clean_person": (d.split == "test") & ~d.leaked & (d.label_by == "person")}
    work.mkdir(parents=True, exist_ok=True)
    for name, sel in lists.items():
        (work / f"{name}.txt").write_text("".join(f"{(DATA / p).as_posix()}\n" for p in d[sel].image))
    yamls = {}
    for name, test in [("clean", "test_clean.txt"), ("clean_person", "test_clean_person.txt")]:
        y = dict(path=DATA.as_posix(), train=(DATA / "yolo_train.txt").as_posix(), val=(work / "val_clean.txt").as_posix(),
                 test=(work / test).as_posix(), names=names)
        yamls[name] = work / f"data_{name}.yaml"
        yamls[name].write_text(yaml.safe_dump(y, sort_keys=False))
    counts = {k: int(v.sum()) for k, v in lists.items()}
    return yamls, names, counts


def evaluate(model, data, names, out, tag):
    m = model.val(data=str(data), split="test", imgsz=model.overrides.get("imgsz", 640), batch=16, plots=True,
                  project=str(out), name=f"eval_{tag}", exist_ok=True, verbose=False)
    b = m.box
    per = {names[c]: dict(ap50=float(b.ap50[i]), ap50_95=float(b.ap[i])) for i, c in enumerate(m.ap_class_index)}
    return dict(map50=float(b.map50), map50_95=float(b.map), precision=float(b.mp), recall=float(b.mr), per_class=per,
                n_images=int(m.seen) if hasattr(m, "seen") else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt", help="pretrained Ultralytics checkpoint to start from")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=15, help="stop when clean-val mAP has not improved for this many epochs")
    p.add_argument("--workers", type=int, default=2, help="data-loading processes; 4 ran out of memory on a 16 GB laptop")
    p.add_argument("--resume", action="store_true", help="continue a crashed run from <model>/weights/last.pt")
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()
    assert LEAKS.exists(), "run models/classifier/scripts/Aiden/find_leaks.py first"
    if sys.platform == "win32":   # keep Windows awake while training
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    from ultralytics import YOLO

    tag = Path(a.model).stem
    out = RESULTS / tag
    yamls, names, counts = write_lists(RESULTS / "_lists")
    print("clean lists:", counts)
    if a.resume:
        YOLO(str(out / "weights" / "last.pt")).train(resume=True, workers=a.workers)
    elif not a.eval_only:
        model = YOLO(a.model)
        model.train(data=str(yamls["clean"]), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
                    workers=a.workers, seed=SEED, deterministic=True, project=str(RESULTS), name=tag, exist_ok=True, plots=True)
    best = out / "weights" / "best.pt"
    model = YOLO(str(best))

    results = dict(full=evaluate(model, DATA / "data.yaml", names, out, "full"),
                   person=evaluate(model, DATA / "data_person.yaml", names, out, "person"),
                   clean=evaluate(model, yamls["clean"], names, out, "clean"),
                   clean_person=evaluate(model, yamls["clean_person"], names, out, "clean_person"))
    hist = pd.read_csv(out / "results.csv")
    hist.columns = hist.columns.str.strip()
    metrics = dict(model=tag, settings=dict(epochs_planned=a.epochs, epochs_run=int(len(hist)), imgsz=a.imgsz, batch=a.batch,
                                            patience=a.patience, seed=SEED), lists=counts, classes=names,
                   best_epoch=int(hist["metrics/mAP50-95(B)"].idxmax() + 1), results=results)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    lines = [f"YOLO {tag}: test results (best.pt, chosen on the clean val set)"]
    for k, label in [("full", "Daniel's full test (compare with groupmates; includes copies of training photos)"),
                     ("person", "Daniel's person-labelled test (data_person.yaml)"),
                     ("clean", f"clean test ({counts['test_clean']} images, no copies of training photos)"),
                     ("clean_person", f"clean + person-labelled test ({counts['test_clean_person']} images)")]:
        r = results[k]
        lines.append(f"\n{label}\n  mAP50 {r['map50']:.3f}  mAP50-95 {r['map50_95']:.3f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}")
        lines += [f"    {c:<20s} AP50 {v['ap50']:.3f}  AP50-95 {v['ap50_95']:.3f}" for c, v in r["per_class"].items()]
    (out / "test_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # team log: one row per member + model, replaced on every run
    f, c = results["full"], results["clean"]
    row = dict(run_id=f"aiden_{tag}", date=date.today().isoformat(), member="Aiden", model=tag,
               data_version=f"oral_dataset 2026-11-11 ({len(pd.read_csv(DATA / 'labels.csv', usecols=['md5']))} images)",
               imgsz=a.imgsz, epochs=len(hist), batch=a.batch, lr0=model.ckpt.get("train_args", {}).get("lr0", "") if model.ckpt else "",
               mAP50=round(f["map50"], 4), mAP50_95=round(f["map50_95"], 4), precision=round(f["precision"], 4), recall=round(f["recall"], 4),
               weights_path=best.relative_to(REPO).as_posix(),
               notes=f"metrics on data.yaml test; leak-free subset ({counts['test_clean']} imgs): mAP50 {c['map50']:.3f} mAP50-95 {c['map50_95']:.3f}")
    log = pd.read_csv(TEAM_LOG) if TEAM_LOG.stat().st_size else pd.DataFrame(columns=list(row))
    log = pd.concat([log[log.run_id != row["run_id"]], pd.DataFrame([row])], ignore_index=True)
    log.to_csv(TEAM_LOG, index=False)

    sys.path.insert(0, str(Path(__file__).parent))
    import report_yolo
    report_yolo.write_all()


if __name__ == "__main__":
    main()
