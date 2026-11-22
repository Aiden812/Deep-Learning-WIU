"""Classifier vs YOLO as the normal/abnormal decision, from clf_vs_yolo_score.py's per-image scores.

Usage: python clf_vs_yolo_analyse.py models/classifier/results/Daniel/binary/clf_vs_yolo/scores.csv

Image-level verdicts: classifier p(issue) >= its checkpoint threshold; YOLO = any box >= best.json conf.
Sets: the shared test split, and the cross-source sets (xsrc_test + xsrc_val) split by whether YOLO
trained on the image (yolo_train.txt includes most xsrc images; the classifier never trained on them).
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import binomtest, rankdata

CT, YT = 0.7285046577453613, 0.361  # classifier threshold (binary/best.pt), YOLO conf (best.json)
d = pd.read_csv(sys.argv[1])
d["c"] = d.p_issue >= CT
d["yv"] = d.yolo_max >= YT
xs = d.role.str.startswith("xsrc")
SETS = [("TEST (in-distribution)", d.role == "test"),
        ("XSRC, YOLO trained on them", xs & d.in_yolo_train),
        ("XSRC, neither model trained on them", xs & ~d.in_yolo_train)]
rng = np.random.default_rng(0)


def sens_at_spec(s, y, spec):
    return (s[y == 1] > np.quantile(s[y == 0], spec)).mean()


for name, mask in SETS:
    s = d[mask]; y = s.has_issue.values
    print(f"\n=== {name}: n={len(s)} issue={y.sum()} normal={(1 - y).sum()}")
    ens = (rankdata(s.p_issue) + rankdata(s.yolo_max)) / 2
    for lab, sc in [("classifier", s.p_issue), ("yolo", s.yolo_max), ("rank-avg", ens)]:
        print(f"{lab:11s} AUROC={roc_auc_score(y, sc):.4f} AP={average_precision_score(y, sc):.4f}")
    diffs = []
    for _ in range(1000):
        i = rng.integers(0, len(s), len(s)); yy = y[i]
        if yy.min() != yy.max():
            diffs.append(roc_auc_score(yy, s.p_issue.values[i]) - roc_auc_score(yy, s.yolo_max.values[i]))
    print("AUROC(clf)-AUROC(yolo) 95% CI", np.round(np.percentile(diffs, [2.5, 50, 97.5]), 4))
    for lab, pred in [("classifier", s.c), ("yolo", s.yv), ("AND", s.c & s.yv), ("OR", s.c | s.yv)]:
        p = pred.values
        print(f"{lab:11s} sens={p[y == 1].mean():.4f} spec={1 - p[y == 0].mean():.4f} FN={(~p[y == 1]).sum()} FP={p[y == 0].sum()}")
    for spec in (0.90, 0.95):
        print(f"sens@spec{spec}: clf={sens_at_spec(s.p_issue.values, y, spec):.3f} yolo={sens_at_spec(s.yolo_max.values, y, spec):.3f}")
    rc, ry = s.c.values == y, s.yv.values == y
    b, c = (rc & ~ry).sum(), (~rc & ry).sum()
    print(f"clf right/yolo wrong={b}, yolo right/clf wrong={c}, McNemar p={binomtest(b, b + c).pvalue if b + c else 1:.3g}")
    iss = s[s.has_issue == 1]
    print(iss.groupby("dataset").agg(n=("image", "size"), clf_sens=("c", "mean"), yolo_sens=("yv", "mean")).round(3).to_string())

# What YOLO conf would match the classifier's recall on cross-source issue photos, and what it costs.
t = d[d.role == "test"]; x = d[xs]
xi = x[(x.has_issue == 1) & x.in_yolo_train]; tn = t[t.has_issue == 0]; xn = x[x.has_issue == 0]
print(f"\nclassifier: xsrc issue sens {(xi.p_issue >= CT).mean():.3f} | test normal FAR {(tn.p_issue >= CT).mean():.3f}"
      f" | xsrc normal FAR {(xn.p_issue >= CT).mean():.3f}")
for thr in (0.361, 0.25, 0.15, 0.10, 0.05, 0.03):
    print(f"yolo conf {thr:.3f}: xsrc issue sens {(xi.yolo_max >= thr).mean():.3f} | test normal FAR "
          f"{(tn.yolo_max >= thr).mean():.3f} | xsrc normal FAR {(xn.yolo_max >= thr).mean():.3f}")

# Per condition (single-condition images), cross-source issue photos YOLO trained on.
L = pd.read_csv("data/raw/oral_dataset/oral_dataset/labels.csv", usecols=["image", "labels"])
xl = xi.merge(L, on="image")
print(xl.groupby("labels").agg(n=("image", "size"), clf_sens=("c", "mean"), yolo_sens=("yv", "mean"))
      .query("n >= 20").round(3).to_string())
k = xi[xi.dataset == "kaggle_oral"]
print(f"kaggle_oral issue: median yolo_max {k.yolo_max.median():.3f}, median p_issue {k.p_issue.median():.3f}")
