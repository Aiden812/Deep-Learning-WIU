"""Retrain Brandon's issue-or-not gate (EfficientNet-B0) so the photo's SOURCE can't give the answer away.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/models/classifier/scripts/Brandon/train_issue_v13.py
    ISSUE_SMOKE=1 ...   (small sample, 2 epochs)

Why: the v1.1/v1.2 gates are near chance on MetaDent DS1 hospital photos (AUROC 0.54-0.58) and flag 35-54% of
normal cheek / tongue photos. In the v1.2 pool, Roboflow photos are all "issue" and Mendeley photos all "healthy",
so "which dataset does this look like" is a strong shortcut.

What changes from 03_issue_classifier_v12.ipynb (same model, image size, cache and checkpoint format):
  * Fresh ImageNet start (not the v1.2 weights, which already learned the shortcut).
  * Extra training photos from the benchmark manifest's TRAIN part only (80% "fit" slice, see models/v13_split.py):
    MetaDent DS3 problem / healthy photos (both labels from one source), Autooral ulcers (issue), SMART-OM normal
    tongue / cheek / lip (healthy). AlphaDent is left out: its no-caries photos aren't checked for calculus or
    discoloration, so they aren't known to be healthy.
  * Source-balanced sampling: each (source, label) cell gets weight proportional to sqrt(its size), then both labels
    get equal total weight. Big single-label sources (Roboflow, Mendeley) no longer dominate an epoch.
  * Best epoch = mean of val_fair AUROC (v1.2 mixed sources) and MetaDent held-out AUROC.
  * Threshold = best mean balanced accuracy over those two val sets (not 95% recall on v1.2 only).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))                                   # Deep-Learning-WIU/models
sys.path.insert(0, str(HERE.parents[2] / "notebooks" / "Brandon"))         # issue_data.py
from issue_data import OralDS, make_cache, make_cache_nopad  # noqa: E402
from v13_split import BENCH_OUT, POOL, REPO, WORKSPACE, external_train_rows, keep_awake  # noqa: E402

SMOKE = os.environ.get("ISSUE_SMOKE") == "1"
EXPERIMENT = "B3_effb0_issue_v13" + ("_smoke" if SMOKE else "")
CACHE_SIDE, IMG_SIZE, BATCH = 288, 224, 32
HEAD_EPOCHS, EPOCHS = 1, (2 if SMOKE else 14)
LR_HEAD, LR, WEIGHT_DECAY, PATIENCE = 1e-3, 3e-4, 1e-4, 4
UNSURE_BELOW = 0.65
NUM_WORKERS = 4
MIXED_SOURCES = ["zenodo", "dentalai", "kaggle_oral", "external_clean"]
SEED = 42
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
POOL_CACHE = POOL.parent / f"cache_{CACHE_SIDE}_nopad"
EXT_CACHE = WORKSPACE / "datasets" / "v13_work" / f"cls_cache_{CACHE_SIDE}"
OUT_DIR = REPO / "models/classifier/results/Brandon"
WEIGHTS = REPO / "models/classifier/weights/Brandon" / ("issue_classifier_v13" + ("_smoke" if SMOKE else "") + ".pt")
CKPT = WORKSPACE / "datasets" / "v13_work" / f"{EXPERIMENT}_last.pt"     # resume point


def load_data():
    pool = pd.read_csv(POOL / "pool.csv")
    pool = pool.assign(src=[str(POOL / p) for p in pool.image],
                       cache=[str(POOL_CACHE / (Path(p).stem + ".jpg")) for p in pool.image],
                       source=pool.dataset, y=pool.has_issue.astype(int), part=pool.split,
                       fair=pool.dataset.isin(MIXED_SOURCES), ext=False)
    ext = external_train_rows()
    keep = ((ext.dataset == "metadent") & ext.problem.notna()) | ext.dataset.isin(["autooral", "smartom"])
    ext = ext[keep & ext.reason.isna()].copy()
    ext["y"] = np.where(ext.dataset == "autooral", 1, np.where(ext.dataset == "smartom", 0, ext.problem.fillna(0))).astype(int)
    ext = ext.assign(src=[str(BENCH_OUT / p) for p in ext.image],
                     cache=[str(EXT_CACHE / (i.replace("/", "__") + ".jpg")) for i in ext.id],
                     source=ext.dataset, part=ext.role13.map({"fit": "train", "val": "val"}), fair=False, ext=True)
    cols = ["src", "cache", "source", "y", "part", "fair", "ext", "orig_w", "orig_h"]
    ext["orig_w"], ext["orig_h"] = ext.width, ext.height
    df = pd.concat([pool[cols], ext[cols]], ignore_index=True)
    if SMOKE:
        df = pd.concat([g.sample(min(len(g), 60), random_state=SEED) for _, g in df.groupby(["part", "source", "y"])])
    EXT_CACHE.mkdir(parents=True, exist_ok=True)
    POOL_CACHE.mkdir(parents=True, exist_ok=True)

    def cache(r):
        if r.ext:
            return make_cache(r.src, r.cache, CACHE_SIDE)
        return make_cache_nopad(r.src, r.cache, CACHE_SIDE, int(r.orig_w), int(r.orig_h))
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        ok = list(ex.map(cache, df.itertuples()))
    df = df[np.array(ok)].reset_index(drop=True)
    print(f"{len(ok) - sum(ok)} unreadable photos dropped")
    return df


def sample_weights(tr):
    n = tr.groupby(["source", "y"]).size()
    cell = np.sqrt(n.astype(float))
    for lab in (0, 1):                                        # equal total weight per label
        cell[cell.index.get_level_values(1) == lab] /= cell[cell.index.get_level_values(1) == lab].sum()
    w = tr.apply(lambda r: cell[(r.source, r.y)] / n[(r.source, r.y)], axis=1).to_numpy()
    share = (cell / cell.groupby(level=1).transform("sum") / 2).rename("share of an epoch")
    print(pd.concat([n.rename("photos"), share.round(3)], axis=1).to_string())
    return torch.as_tensor(w, dtype=torch.double)


def bal_at(y, p, t):
    f = p >= t
    return ((f[y == 1]).mean() + (~f[y == 0]).mean()) / 2


def main():
    keep_awake()
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data()
    tr = df[df.part == "train"].reset_index(drop=True)
    va = df[df.part == "val"].reset_index(drop=True)
    te = df[df.part == "test"].reset_index(drop=True)
    vf = va[va.fair].reset_index(drop=True)                      # v1.2 val_fair
    vm = va[va.source == "metadent"].reset_index(drop=True)      # held-out MetaDent
    print(f"train {len(tr)} | val {len(va)} (val_fair {len(vf)}, MetaDent held-out {len(vm)}) | test {len(te)}")

    train_tf = v2.Compose([v2.ToImage(), v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33), antialias=True),
                           v2.RandomHorizontalFlip(), v2.RandomRotation(15),
                           v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                           v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
    eval_tf = v2.Compose([v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
                          v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def loader(frame, tf, sampler=None):
        return torch.utils.data.DataLoader(OralDS(frame.cache, frame.y, tf), batch_size=BATCH, sampler=sampler,
                                           num_workers=NUM_WORKERS, pin_memory=dev.type == "cuda",
                                           persistent_workers=NUM_WORKERS > 0)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights(tr), num_samples=len(tr), replacement=True)
    train_dl = loader(tr, train_tf, sampler)
    dls = {k: loader(f, eval_tf) for k, f in dict(val=va, val_fair=vf, val_md=vm, test=te).items()}

    m = tv.models.efficientnet_b0(weights=tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m = m.to(dev)
    crit = nn.CrossEntropyLoss()           # the sampler already balances the labels

    @torch.no_grad()
    def logits_of(dl):
        m.eval(); out, ys = [], []
        for x, y in dl:
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                out.append(m(x.to(dev, non_blocking=True)).float().cpu())
            ys.append(y)
        return torch.cat(out), torch.cat(ys).numpy()

    def phase(head_only):
        for p in m.features.parameters():
            p.requires_grad = not head_only
        return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR_HEAD if head_only else LR,
                                 weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    start, hist, best, best_state, bad = 1, [], -1.0, None, 0
    opt, sched = phase(True), None
    if CKPT.exists():                       # resume after an interruption
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"]); start, hist, best, best_state, bad = ck["epoch"] + 1, ck["hist"], ck["best"], ck["best_state"], ck["bad"]
        if start > HEAD_EPOCHS:
            opt = phase(False)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, (EPOCHS - HEAD_EPOCHS) * len(train_dl)))
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        print(f"resuming at epoch {start}")

    for ep in range(start, EPOCHS + 1):
        if bad >= PATIENCE and ep > HEAD_EPOCHS + 1:
            break
        if ep == HEAD_EPOCHS + 1:
            opt = phase(False)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, (EPOCHS - HEAD_EPOCHS) * len(train_dl)))
        m.train(); t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = crit(m(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched is not None:
                sched.step()
            tot += loss.item() * len(y); n += len(y)
        lf, yf = logits_of(dls["val_fair"]); lm, ym = logits_of(dls["val_md"])
        auc_f = roc_auc_score(yf, lf.softmax(1)[:, 1]); auc_m = roc_auc_score(ym, lm.softmax(1)[:, 1])
        score = (auc_f + auc_m) / 2
        hist.append(dict(epoch=ep, train_loss=tot / n, val_fair_auc=auc_f, val_metadent_auc=auc_m, score=score,
                         minutes=(time.time() - t0) / 60))
        print(f"epoch {ep}: loss {tot / n:.3f} | val_fair AUROC {auc_f:.3f} | MetaDent held-out AUROC {auc_m:.3f} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
        torch.save(dict(model=m.state_dict(), epoch=ep, hist=hist, best=best, best_state=best_state, bad=bad,
                        opt=opt.state_dict(), sched=sched.state_dict() if sched else None), CKPT)

    m.load_state_dict(best_state)
    best_epoch = max(hist, key=lambda h: h["score"])["epoch"]
    # temperature on the whole val set, then the threshold
    lv, yv = logits_of(dls["val"])
    log_T = torch.zeros(1, requires_grad=True)
    o = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        o.zero_grad(); l_ = nn.functional.cross_entropy(lv / log_T.exp(), torch.as_tensor(yv)); l_.backward(); return l_
    o.step(closure)
    T = float(log_T.exp())
    pf = (logits_of(dls["val_fair"])[0] / T).softmax(1)[:, 1].numpy()
    pm = (logits_of(dls["val_md"])[0] / T).softmax(1)[:, 1].numpy()
    cuts = np.unique(np.r_[pf, pm])
    vals = [(bal_at(yf, pf, c) + bal_at(ym, pm, c)) / 2 for c in cuts]
    thr = float(cuts[int(np.argmax(vals))])
    lt, yt = logits_of(dls["test"])
    pt = (lt / T).softmax(1)[:, 1].numpy()
    fair = te.fair.to_numpy()
    res = dict(threshold=thr, temperature=T, best_epoch=best_epoch,
               val_fair=dict(auc=float(roc_auc_score(yf, pf)), bal_acc=float(bal_at(yf, pf, thr))),
               val_metadent=dict(auc=float(roc_auc_score(ym, pm)), bal_acc=float(bal_at(ym, pm, thr)), n=len(ym)),
               test_fair=dict(auc=float(roc_auc_score(yt[fair], pt[fair])), bal_acc=float(bal_at(yt[fair], pt[fair], thr))),
               test_all=dict(auc=float(roc_auc_score(yt, pt)), bal_acc=float(bal_at(yt, pt, thr))))
    print(json.dumps(res, indent=1))

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": "issue_or_not", "backbone": "efficientnet_b0", "head": "classifier.1", "classes": ["healthy", "issue"],
                "state_dict": best_state, "img_size": IMG_SIZE, "mean": MEAN, "std": STD, "temperature": T,
                "threshold": thr, "unsure_below": UNSURE_BELOW,
                "config": {"cache_max_side": CACHE_SIDE, "padding_cut_off": True, "epochs_run": len(hist),
                           "best_epoch": best_epoch, "batch": BATCH, "lr": LR, "seed": SEED,
                           "dataset": "oral_datasetv1.2 pool + benchmark train parts (MetaDent DS3, Autooral, SMART-OM), "
                                      "source-balanced sampling"},
                "notes": "Screening aid only, not a diagnosis. v1.3: trained with source-balanced sampling and external "
                         "photos so the photo's source is a weaker clue."}, WEIGHTS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{EXPERIMENT}_metrics.json").write_text(json.dumps(dict(experiment=EXPERIMENT, **res), indent=1))
    pd.DataFrame(hist).to_csv(OUT_DIR / f"{EXPERIMENT}_history.csv", index=False)
    CKPT.unlink(missing_ok=True)
    print("saved", WEIGHTS, flush=True)


if __name__ == "__main__":
    main()
