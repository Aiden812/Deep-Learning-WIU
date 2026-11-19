"""Step 5: score every model that has predictions and write the report.

    .venv-train/Scripts/python.exe Deep-Learning-WIU/benchmark/score.py

Writes benchmark/results/benchmark_report.md and results.json. Primary numbers use role == bench only;
"all" rows add the train part (still unseen by today's models, NOT valid for models trained on it later).
95% CIs: 1,000 bootstrap resamples of whole groups (AlphaDent / Autooral / SMART-OM patients, MetaDent
near-duplicate clusters). Thresholds are each model's own; nothing is tuned here.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import MODELS  # noqa: E402
from common import CLASS_COL, CLASSES, MANIFEST, OUT, PRED_DIR, RESULTS, SEED  # noqa: E402

B = 1000
rng = np.random.default_rng(SEED)
DET_COL = {c: f"p_{c}" for c in CLASSES}


# ---- bootstrap over groups ---------------------------------------------------------------------------------------
def boot(df, fn, n=B):
    """point estimate and 95% CI of fn(df) with whole groups resampled."""
    est = fn(df)
    if est is None or np.isnan(est):
        return est, None, None
    idx = df.groupby("group").indices
    keys = list(idx)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        v = fn(df.iloc[np.concatenate([idx[keys[i]] for i in pick])])
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < n / 2:
        return est, None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, lo, hi


def auroc(y, s):
    return lambda d: roc_auc_score(d[y], d[s]) if d[y].nunique() == 2 else np.nan


def rate(cond_col, cond_val, flag_col):
    def f(d):
        g = d[d[cond_col] == cond_val]
        return np.nan if g.empty else float((g[flag_col].astype(bool) == bool(cond_val)).mean())
    return f


def bal(y, flag):
    def f(d):
        a, b = rate(y, 1, flag)(d), rate(y, 0, flag)(d)
        return np.nan if np.isnan(a) or np.isnan(b) else (a + b) / 2
    return f


def fmt(t, pct=False):
    est, lo, hi = t
    if est is None or (isinstance(est, float) and np.isnan(est)):
        return "–"
    k = 100 if pct else 1
    s = f"{est * k:.0f}%" if pct else f"{est:.3f}"
    return s + (f" ({lo * k:.0f}–{hi * k:.0f})" if pct and lo is not None else f" ({lo:.2f}–{hi:.2f})" if lo is not None else "")


# ---- box AP50 (all-point interpolation, greedy matching per photo) ---------------------------------------------
def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-12)


def read_gt(label_rel, cls):
    p = OUT / label_rel
    if not isinstance(label_rel, str) or not p.exists():
        return np.zeros((0, 4))
    v = [list(map(float, l.split())) for l in p.read_text().splitlines() if l.strip()]
    v = np.array([r[1:] for r in v if int(r[0]) == cls]).reshape(-1, 4)
    return np.c_[v[:, 0] - v[:, 2] / 2, v[:, 1] - v[:, 3] / 2, v[:, 0] + v[:, 2] / 2, v[:, 1] + v[:, 3] / 2]


def match(pred_json, gt, cls, thr=0.5):
    """-> (confs, tp flags) for one photo; a prediction is a hit when its IoU with an unused truth box >= thr."""
    boxes = [b for b in json.loads(pred_json) if b[0] == cls] if isinstance(pred_json, str) else []
    boxes.sort(key=lambda b: -b[1])
    used = np.zeros(len(gt), bool)
    conf, tp = [], []
    for b in boxes:
        ok = False
        if len(gt):
            o = iou(np.array(b[2:]), gt)
            o[used] = 0
            j = int(o.argmax())
            if o[j] >= thr:
                used[j] = ok = True
        conf.append(b[1]); tp.append(ok)
    return np.array(conf), np.array(tp, bool)


def ap_from(conf, tp, n_gt):
    if n_gt == 0:
        return np.nan
    o = np.argsort(-conf)
    tpc = np.cumsum(tp[o]); fpc = np.cumsum(~tp[o])
    rec = np.r_[0, tpc / n_gt, 1]; prec = np.r_[1, tpc / np.maximum(tpc + fpc, 1), 0]
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    i = np.nonzero(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[i + 1] - rec[i]) * prec[i + 1]))


def box_track(d, cls, conf_thr):
    """AP at IoU 0.5 (strict) and 0.1 (lesion found, box size ignored) + recall/precision at the model's own conf,
    with group-bootstrap CIs."""
    gts = [read_gt(lab, cls) for lab in d.label]
    ngt = np.array([len(g) for g in gts])
    d = d.assign(_i=np.arange(len(d)))
    out = dict(n_photos=len(d), n_boxes=int(ngt.sum()))
    for thr, tag in [(0.5, "50"), (0.1, "10")]:
        per = [match(pj, g, cls, thr) for pj, g in zip(d.boxes, gts)]
        hits = np.array([int((t & (c >= conf_thr)).sum()) for c, t in per])
        npred = np.array([int((c >= conf_thr).sum()) for c, _ in per])

        def ap(x, per=per):
            ii = x._i.to_numpy()
            return ap_from(np.concatenate([per[i][0] for i in ii]), np.concatenate([per[i][1] for i in ii]), ngt[ii].sum())

        out[f"AP{tag}"] = boot(d, ap)
        out[f"recall{tag}"] = boot(d, lambda x, h=hits: h[x._i].sum() / max(1, ngt[x._i].sum()))
        out[f"precision{tag}"] = boot(d, lambda x, h=hits, n=npred: np.nan if n[x._i].sum() == 0 else h[x._i].sum() / n[x._i].sum())
    return out


# ---- cheat check -------------------------------------------------------------------------------------------------
def size_only(d):
    x = np.c_[d.width, d.height, d.width / d.height]
    y = d.problem.astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return np.nan
    p = cross_val_predict(DecisionTreeClassifier(max_depth=3, random_state=SEED), x, y, cv=5)
    return np.mean([(p[y == 1] == 1).mean(), (p[y == 0] == 0).mean()])


# ------------------------------------------------------------------------------------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    man = pd.read_csv(MANIFEST, low_memory=False)
    preds = {f.stem: pd.read_csv(f) for f in sorted(PRED_DIR.glob("*.csv")) if f.stem in MODELS}
    if not preds:
        sys.exit("no predictions yet: run run.py first")
    speed = pd.read_csv(PRED_DIR / "speed.csv").set_index("model") if (PRED_DIR / "speed.csv").exists() else None
    R, md = {}, []
    md.append(f"# External benchmark results ({time.strftime('%Y-%m-%d')})\n")
    md.append("Photos no team model trained on (copies removed). Primary = **bench** part. 95% CIs in brackets "
              "(bootstrap over patients / photo clusters). Each model uses its **own** saved threshold.\n")
    md.append("| model | owner | what | photos predicted |\n|---|---|---|---|")
    for m, p in preds.items():
        md.append(f"| {m} | {MODELS[m]['owner']} | {MODELS[m]['note']} | {len(p)} |")
    missing = [m for m in MODELS if m not in preds]
    if missing:
        md.append(f"\nNot scored yet (no weights on this laptop): {', '.join(missing)}.\n")

    J = {m: man.merge(p, on="id") for m, p in preds.items()}

    # ---- Track 1: problem vs healthy (MetaDent) -----------------------------------------------------------------
    md.append("\n## 1. Problem vs healthy (MetaDent)\n")
    md.append("Problem = caries, plaque/calculus, discoloration, missing teeth or ulcer; healthy = no finding at all. "
              "Photos whose only findings are outside our 5 conditions (gums, wear, braces, fillings) are left out. "
              "**DS1 = hospital photos (fair: same camera for both groups); DS3 = web photos** (healthy ones are often "
              "stock smiles, so style can give the answer away).\n")
    for sub in ["DS1", "DS3"]:
        base = man[(man.dataset == "metadent") & (man.role == "bench") & (man.subset == sub) & man.problem.notna()]
        md.append(f"\n### {sub}: {int((base.problem == 1).sum())} problem / {int((base.problem == 0).sum())} healthy photos; "
                  f"size-only cheat tree balanced acc {size_only(base):.2f}\n")
        md.append("| model | AUROC | balanced acc | sensitivity | specificity |\n|---|---|---|---|---|")
        for m, d in J.items():
            d = d[(d.dataset == "metadent") & (d.role == "bench") & (d.subset == sub) & d.problem.notna()]
            if d.p_problem.isna().all():
                continue
            r = dict(auroc=boot(d, auroc("problem", "p_problem")), bal=boot(d, bal("problem", "flag_problem")),
                     sens=boot(d, rate("problem", 1, "flag_problem")), spec=boot(d, rate("problem", 0, "flag_problem")))
            R.setdefault(m, {})[f"t1_{sub}"] = r
            md.append(f"| {m} | {fmt(r['auroc'])} | {fmt(r['bal'], True)} | {fmt(r['sens'], True)} | {fmt(r['spec'], True)} |")

    # ---- Track 2: per condition, photo level ---------------------------------------------------------------------
    md.append("\n## 2. Each condition, photo level\n")
    md.append("Models that name conditions: AUROC of that condition's score (photos with it vs photos without it). "
              "Gates (one yes/no score): share of photos with that condition they flag. MetaDent = DS1 + DS3 bench; "
              "AlphaDent caries = dentist-checked photos with / without caries (both from clinic patients).\n")
    md.append("| model | " + " | ".join(["Caries (AlphaDent)", "Caries", "Calculus/plaque", "Discoloration", "Missing teeth"]) + " |")
    md.append("|---|---|---|---|---|---|")
    for m, d in J.items():
        cells = []
        for ds, c in [("alphadent", "Caries"), ("metadent", "Caries"), ("metadent", "Calculus"),
                      ("metadent", "Tooth_Discoloration"), ("metadent", "Hypodontia")]:
            col = CLASS_COL[c]
            x = d[(d.dataset == ds) & (d.role == "bench") & d[col].notna()]
            if x[f"p_{c}"].notna().any():
                r = boot(x, auroc(col, f"p_{c}"))
                cells.append(f"AUROC {fmt(r)}")
            else:
                r = boot(x, rate(col, 1, "flag_problem"))
                cells.append(f"flags {fmt(r, True)}")
            R.setdefault(m, {})[f"t2_{ds}_{c}"] = r
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md.append("\nPositives in the bench part: " + ", ".join(
        f"{lbl} {int(((man.dataset == ds) & (man.role == 'bench') & (man[col] == 1)).sum())}/"
        f"{int(((man.dataset == ds) & (man.role == 'bench') & man[col].notna()).sum())}"
        for lbl, ds, col in [("AlphaDent caries", "alphadent", "caries"), ("caries", "metadent", "caries"),
                             ("calculus/plaque", "metadent", "calculus"), ("discoloration", "metadent", "discoloration"),
                             ("missing teeth", "metadent", "missing_teeth")]) + " photos.")

    # ---- Track 3: caries boxes (AlphaDent) -----------------------------------------------------------------------
    md.append("\n## 3. Caries boxes (AlphaDent, dentist outlines)\n")
    md.append("Dentist polygons for caries classes 1-6 merged into one Caries box each; abrasion, fillings and crowns "
              "are not boxed. AlphaDent outlines **only the lesion**, while our training boxes (Roboflow style) often "
              "cover the whole tooth, so the strict score (IoU >= 0.5) punishes box size as well as missed caries. "
              "The loose score (IoU >= 0.1) asks only \"did a caries box land on the lesion?\". Recall / precision at "
              "the model's own confidence cut.\n")
    md.append("| model | part | photos | caries lesions | AP50 (strict) | AP10 (loose) | lesions found (loose) | boxes on a lesion (loose) |\n|---|---|---|---|---|---|---|---|")
    for m, d in J.items():
        if d.boxes.isna().all():
            continue
        conf = d.attrs.get("conf", MODELS[m].get("conf", 0.25))
        for part, roles in [("bench", ["bench"]), ("all", ["bench", "train"])]:
            x = d[(d.dataset == "alphadent") & d.role.isin(roles)]
            r = box_track(x, 1, conf)
            R.setdefault(m, {})[f"t3_{part}"] = r
            md.append(f"| {m} | {part} | {r['n_photos']} | {r['n_boxes']} | {fmt(r['AP50'])} | {fmt(r['AP10'])} | "
                      f"{fmt(r['recall10'], True)} | {fmt(r['precision10'], True)} |")

    # ---- Track 4: ulcers -----------------------------------------------------------------------------------------
    md.append("\n## 4. Mouth ulcers\n")
    md.append("Autooral: ulcer close-ups with dentist masks (every photo has an ulcer). SMART-OM: normal tongue, cheek "
              "and lip photos (no ulcer), scored **separately** because the two sets look different. A gate counts as "
              "catching an ulcer if it flags the photo at all.\n")
    md.append("| model | ulcer photos caught (Autooral) | ulcer box AP50 | normal soft-tissue photos wrongly flagged (SMART-OM) |\n|---|---|---|---|")
    for m, d in J.items():
        ao = d[(d.dataset == "autooral") & (d.role == "bench")]
        om = d[(d.dataset == "smartom") & (d.role == "bench")]
        fl = "flag_Mouth_Ulcer" if ao["flag_Mouth_Ulcer"].notna().any() else "flag_problem"
        caught = boot(ao, lambda x: float(x[fl].astype(bool).mean()))
        false = boot(om, lambda x: float(x[fl].astype(bool).mean()))
        ap = box_track(ao, 3, MODELS[m].get("conf", 0.25))["AP50"] if ao.boxes.notna().any() else (np.nan, None, None)
        R.setdefault(m, {})["t4"] = dict(caught=caught, false_alarm=false, ap50=ap, by=fl)
        tag = "" if fl == "flag_Mouth_Ulcer" else " (any-problem flag)"
        md.append(f"| {m} | {fmt(caught, True)}{tag} | {fmt(ap)} | {fmt(false, True)}{tag} |")
    ao_b = man[(man.dataset == "autooral") & (man.role == "bench")]
    om_b = man[(man.dataset == "smartom") & (man.role == "bench")]
    md.append(f"\n{len(ao_b)} Autooral photos ({ao_b.group.nunique()} patients), {len(om_b)} SMART-OM photos "
              f"({om_b.group.nunique()} patients).")

    # ---- Track 5: speed --------------------------------------------------------------------------------------------
    if speed is not None:
        md.append("\n## 5. Speed and size (CPU, one photo at a time)\n")
        md.append("| model | median ms/photo | 90th pct ms | weights MB |\n|---|---|---|---|")
        for m in preds:
            if m in speed.index:
                s = speed.loc[m]
                md.append(f"| {m} | {s.cpu_ms_median:.0f} | {s.cpu_ms_p90:.0f} | {s.size_mb:.1f} |")

    # ---- data summary --------------------------------------------------------------------------------------------
    md.append("\n## Data used\n")
    t = man.groupby(["dataset", "role"]).size().unstack(fill_value=0)
    md.append("| dataset | " + " | ".join(t.columns) + " |\n|---|" + "---|" * len(t.columns))
    for ds, row in t.iterrows():
        md.append(f"| {ds} | " + " | ".join(str(v) for v in row) + " |")
    ex = man[man.role == "excluded"].reason.str.split(" \\(").str[0].value_counts()
    md.append("\nExcluded: " + "; ".join(f"{k}: {v}" for k, v in ex.items()) + ".")
    (RESULTS / "benchmark_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: None if o is None else float(o)))
    print("\n".join(md))


if __name__ == "__main__":
    main()
