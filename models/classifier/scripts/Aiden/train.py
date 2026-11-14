"""Train the stage-1 gate: is this mouth healthy, or does it have an issue? (DenseNet-121 or DINOv2 ViT-B)

    python models/classifier/scripts/Aiden/find_leaks.py              # once: marks val/test copies of training photos
    python models/classifier/scripts/Aiden/train.py --arch densenet121 --lr 1e-4
    python models/classifier/scripts/Aiden/train.py --arch dinov2_vitb14

Pipeline: this binary classifier screens every photo; only photos it flags go on to the YOLO detector (stage 2), which
says which condition and where. A photo the gate misses never reaches YOLO, so the threshold is set for sensitivity: the
highest threshold that still catches TARGET_SENSITIVITY of the issues in the clean val set.

Data: data/raw/oral_dataset (oral_dataset.zip from Daniel's qwen_clean_and_label.ipynb), target `has_issue`. Training
uses Daniel's train split as it is. Val and test are his too, but only the images with no copy in train are used for the
threshold and the headline score ("clean"); the score on his full test set is reported alongside to show how much the
copies inflate it (see find_leaks.py).

Each model has one folder, results/Aiden/<arch>/, overwritten by every run (model.pt, metrics.json, test_report.txt,
train_log.txt, val_predictions.csv, test_predictions.csv). After each run report.py rewrites the single comparison
report and graphs in results/Aiden/.
"""
import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/classifier/src"))
from multilabel import NORM, Net, eval_tf   # noqa: E402

DATA = REPO / "data/raw/oral_dataset"
LEAKS = REPO / "data/processed/oral_dataset_leaks.csv"
RESULTS = REPO / "models/classifier/results/Aiden"
TARGET = "has_issue"
CONDITIONS = ["Calculus", "Caries", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]   # for the per-condition breakdown
NAMES = {"densenet121": "DenseNet-121", "dinov2_vitb14": "DINOv2 ViT-B/14", "dinov2_vits14": "DINOv2 ViT-S/14"}
SEED, DEV = 42, "cuda"
HEAD_LR, SMOOTH = 1e-3, 0.05
TARGET_SENSITIVITY = 0.95


class DS(Dataset):
    def __init__(self, files, y, tf, img):
        self.files, self.y, self.tf, self.img = list(files), torch.tensor(np.asarray(y), dtype=torch.float32), tf, img

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i])
        im.draft("RGB", (2 * self.img, 2 * self.img))   # JPEG: decode at reduced size, much faster for large photos
        return self.tf(im.convert("RGB")), self.y[i]


def param_groups(m, lr, decay=0.75):
    if m.arch == "densenet121":
        return [dict(params=m.body.parameters(), lr=lr), dict(params=m.head.parameters(), lr=HEAD_LR)]
    # ViT: lower layers hold general features, so each block gets a smaller learning rate than the one above it
    blocks = list(m.body.blocks)
    groups = [dict(params=b.parameters(), lr=lr * decay ** (len(blocks) - 1 - i)) for i, b in enumerate(blocks)]
    seen = {id(p) for b in blocks for p in b.parameters()}
    groups.append(dict(params=[p for p in m.body.parameters() if id(p) not in seen], lr=lr * decay ** len(blocks)))
    return groups + [dict(params=m.head.parameters(), lr=HEAD_LR)]


@torch.no_grad()
def logits(model, files, a):
    dl = DataLoader(DS(files, np.zeros(len(files)), eval_tf(a.img), a.img), 64, num_workers=a.workers)
    model.eval()
    out = []
    for x, _ in dl:
        x = x.to(DEV)
        with torch.autocast("cuda"):
            out.append(((model(x) + model(x.flip(3))) / 2).float().cpu())   # same flip averaging as multilabel.ensemble_logits
    return torch.cat(out).numpy()[:, 0]


def ci95(fn, y, z, n=1000):
    """Bootstrap over images: how much the score would move with a different sample of test photos."""
    rng, vals = np.random.default_rng(SEED), []
    while len(vals) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(fn(y[i], z[i]))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def train(frame, a, log, held):
    """Fixed number of epochs (no epoch picked on validation). held = (files, labels), scored after each epoch for the
    training curves only."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(a.img, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),   # no hue/saturation: discolouration is a colour
        transforms.ToTensor(), NORM])
    # every photo group (copies of one photo) equally likely, so a photo with 200 copies doesn't outvote the rest
    copies = frame.group.map(frame.group.value_counts()).to_numpy()
    groups = frame.drop_duplicates("group")
    sampler = WeightedRandomSampler(torch.tensor(1.0 / copies, dtype=torch.double), int(a.passes * len(groups)), replacement=True)
    dl = DataLoader(DS(DATA / frame.image, frame[[TARGET]].values, train_tf, a.img), a.batch, sampler=sampler,
                    num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    pos = groups[TARGET].sum()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(groups) - pos) / pos], dtype=torch.float32, device=DEV))
    model = Net(a.arch, 1).to(DEV)
    pg = param_groups(model, a.lr)
    opt = torch.optim.AdamW(pg, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in pg], total_steps=a.epochs * len(dl),
                                                pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler()
    history = []
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV) * (1 - 2 * SMOOTH) + SMOOTH   # smoothing: some labels come from Qwen and may be wrong
            with torch.autocast("cuda"):
                loss = crit(model(x).float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        z = logits(model, held[0], a)
        h = dict(epoch=ep, train_loss=total / len(dl),
                 heldout_loss=crit(torch.from_numpy(z).to(DEV)[:, None], torch.as_tensor(held[1], dtype=torch.float32, device=DEV)[:, None]).item(),
                 heldout_auroc=float(roc_auc_score(held[1], z)))
        history.append(h)
        log(f"  epoch {ep:2d}  train loss {h['train_loss']:.3f}  clean-val loss {h['heldout_loss']:.3f}  clean-val AUROC {h['heldout_auroc']:.4f}")
    return model.eval(), history


def gate_metrics(y, p, thr):
    flag = p >= thr
    tp, fn, fp, tn = (flag & y).sum(), (~flag & y).sum(), (flag & ~y).sum(), (~flag & ~y).sum()
    return dict(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                precision=float(tp / max(tp + fp, 1)), accuracy=float((tp + tn) / len(y)),
                confusion=dict(tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="dinov2_vitb14")
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5, help="backbone learning rate (top block for ViT)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--passes", type=float, default=1, help="draws per epoch = passes x training photo groups")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    assert LEAKS.exists(), "run find_leaks.py first"
    if sys.platform == "win32":   # keep Windows awake while training
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    out = RESULTS / a.arch
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_log.txt").unlink(missing_ok=True)

    def log(line):
        print(line, flush=True)
        with open(out / "train_log.txt", "a", encoding="utf-8") as f:
            print(line, file=f)

    df = pd.read_csv(DATA / "labels.csv", keep_default_na=False)
    df = df.merge(pd.read_csv(LEAKS)[["md5", "leaked"]], on="md5", how="left")
    df["leaked"] = df.leaked.fillna(False).astype(bool)
    tr = df[df.split == "train"]
    val = df[(df.split == "val") & ~df.leaked]
    test_full = df[df.split == "test"]
    test = test_full[~test_full.leaked]
    log(f"{vars(a)}\ntrain {len(tr)} images ({tr.group.nunique()} photo groups, {int(tr[TARGET].sum())} with an issue) | "
        f"clean val {len(val)} of {(df.split == 'val').sum()} | clean test {len(test)} of {len(test_full)} "
        f"({int(test[TARGET].sum())} with an issue, {int((test[TARGET] == 0).sum())} healthy)")

    model, history = train(tr, a, log, held=(DATA / val.image, val[TARGET].values))
    z_val = logits(model, DATA / val.image, a)
    z_full = logits(model, DATA / test_full.image, a)
    z = z_full[~test_full.leaked.to_numpy()]

    # Calibration (logit -> probability) and the threshold, both from the clean val set only
    cal = LogisticRegression().fit(z_val[:, None], val[TARGET].values)
    prob = lambda v: cal.predict_proba(v[:, None])[:, 1]
    p_val, p_test, p_full = prob(z_val), prob(z), prob(z_full)
    y_val = val[TARGET].to_numpy() == 1
    # highest threshold that still catches TARGET_SENSITIVITY of the clean-val issues
    thr = float(np.sort(p_val[y_val])[int(np.floor((1 - TARGET_SENSITIVITY) * y_val.sum()))])

    y, y_full = test[TARGET].to_numpy() == 1, test_full[TARGET].to_numpy() == 1
    person = (test.label_by == "person").to_numpy()
    flag = p_test >= thr
    m = gate_metrics(y, p_test, thr)
    metrics = dict(
        arch=a.arch, name=NAMES.get(a.arch, a.arch), task="binary gate: has_issue",
        settings=dict(img=a.img, epochs=a.epochs, lr=a.lr, batch=a.batch, passes=a.passes, target_sensitivity=TARGET_SENSITIVITY),
        n_train=len(tr), n_val=len(val), n_test=len(test), n_test_full=len(test_full),
        n_issue=int(y.sum()), n_healthy=int((~y).sum()), threshold=thr, **m,
        auroc_ci=ci95(roc_auc_score, y, p_test), sensitivity_ci=ci95(lambda yy, pp: float(((pp >= thr) & yy).sum() / yy.sum()), y, p_test),
        val=gate_metrics(y_val, p_val, thr),
        person=gate_metrics(y[person], p_test[person], thr),
        full_test=gate_metrics(y_full, p_full, thr),
        # share of images with each condition that the gate passes on to YOLO (a miss here is a miss for the whole pipeline)
        passed_to_yolo={c: dict(n=int(test[c].sum()), share=float(flag[test[c].to_numpy() == 1].mean())) for c in CONDITIONS},
        history=[history])

    def line(name, g):
        return (f"{name:<28s} AUROC {g['auroc']:.4f}  AP {g['ap']:.4f}  sensitivity {g['sensitivity']:.3f}  "
                f"specificity {g['specificity']:.3f}  precision {g['precision']:.3f}  accuracy {g['accuracy']:.3f}")
    report = (f"STAGE-1 GATE: has an issue or healthy. Threshold {thr:.3f} (catches {TARGET_SENSITIVITY:.0%} of clean-val issues)\n\n"
              + line(f"clean test ({len(test)})", m)
              + f"\n{'':<28s} AUROC 95% CI {metrics['auroc_ci'][0]:.4f}-{metrics['auroc_ci'][1]:.4f}, sensitivity 95% CI "
                f"{metrics['sensitivity_ci'][0]:.3f}-{metrics['sensitivity_ci'][1]:.3f}"
              + f"\n{'':<28s} confusion: {m['confusion']}\n"
              + line(f"person-labelled only ({int(person.sum())})", metrics["person"]) + "\n"
              + line(f"full test, leaky ({len(test_full)})", metrics["full_test"]) + "\n"
              + line(f"clean val ({len(val)})", metrics["val"])
              + "\n\npassed on to YOLO, per condition (clean test): "
              + ", ".join(f"{c} {v['share']:.1%} of {v['n']}" for c, v in metrics["passed_to_yolo"].items()))
    log(report)
    (out / "test_report.txt").write_text(report, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))   # read by report.py and prototype/Aiden
    for name, frame, v in (("val", val, p_val), ("test", test, p_test)):
        frame[["image", "label_by", "dataset", TARGET, *CONDITIONS]].assign(p_issue=v).to_csv(out / f"{name}_predictions.csv", index=False)
    torch.save(dict(state_dicts=[model.half().state_dict()], arch=a.arch, img_size=a.img, classes=["has_issue"], thresholds=[thr],
                    calib_a=[float(cal.coef_[0, 0])], calib_b=[float(cal.intercept_[0])], multilabel=False, task="gate"),
               out / "model.pt")
    log(f"saved {out / 'model.pt'}")
    import report as comparison
    comparison.write_all()


if __name__ == "__main__":
    main()
