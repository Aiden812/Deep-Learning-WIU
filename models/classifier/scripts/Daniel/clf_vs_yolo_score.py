"""Score every test / cross-source image with the binary classifier gate and the YOLO detector."""
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision as tv
from PIL import Image, ImageOps
from torchvision.transforms import v2

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

dev = torch.device("cuda")
m = pd.read_csv(REPO / "manifests/Daniel/binary_classifier_split.csv")
m = m[m.role.isin(["test", "xsrc_test", "xsrc_val"])].copy()
ytrain = {os.path.basename(x) for x in open(REPO / "data/raw/oral_dataset/oral_dataset/yolo_train.txt").read().split()}
m["in_yolo_train"] = m.image.map(os.path.basename).isin(ytrain)

# classifier (mirrors demo app.py)
ck = torch.load(REPO / "models/classifier/weights/Daniel/binary/best.pt", map_location="cpu", weights_only=False)
heads = {"convnext": "classifier.2", "efficientnet": "classifier.1", "resnet": "fc", "mobilenet_v3": "classifier.3",
         "densenet": "classifier", "swin": "head"}
name = ck["backbone"]; head = ck.get("head") or next(h for p, h in heads.items() if name.startswith(p))
net = tv.models.get_model(name, weights=None)
pp, _, a = head.rpartition("."); par = net.get_submodule(pp) if pp else net
old = par[int(a)] if a.isdigit() else getattr(par, a)
new = nn.Linear(old.in_features, len(ck["classes"]))
if a.isdigit(): par[int(a)] = new
else: setattr(par, a, new)
net.load_state_dict(ck["state_dict"]); net.to(dev).eval()
T = float(ck.get("temperature", 1.0)); S = int(ck.get("img_size", 224))
cms = int((ck.get("config") or {}).get("cache_max_side", 384))
pos = ck["classes"].index(ck["positive_class"])
tf = v2.Compose([v2.ToImage(), v2.Resize((S, S), antialias=True), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize(tuple(ck.get("mean", (0.485, 0.456, 0.406))), tuple(ck.get("std", (0.229, 0.224, 0.225))))])
print("classifier", name, "T", T, "thr", ck.get("threshold"), "classes", ck["classes"], flush=True)

ymeta = json.loads((REPO / "models/yolo_caries/weights/Daniel/best.json").read_text())
yolo = YOLO(str(REPO / "models/yolo_caries/weights/Daniel/best.pt"), task="detect")
print("yolo", ymeta, flush=True)

rows = []; tc = ty = 0.0
for i, r in enumerate(m.itertuples()):
    im = ImageOps.exif_transpose(Image.open(REPO / r.dataset_dir / r.image)).convert("RGB")
    im.thumbnail((1280, 1280), Image.LANCZOS)
    small = im.copy(); small.thumbnail((cms, cms), Image.LANCZOS)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        p = (net(tf(small)[None].to(dev)).float() / T).softmax(1)[0, pos].item()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    res = yolo.predict(im, imgsz=ymeta["imgsz"], conf=0.001, iou=ymeta["iou"], device=0, verbose=False)[0]
    torch.cuda.synchronize(); t2 = time.perf_counter()
    if i >= 20: tc += t1 - t0; ty += t2 - t1
    conf = res.boxes.conf.cpu().numpy(); cls = res.boxes.cls.cpu().numpy().astype(int)
    d = {"image": r.image, "p_issue": p, "yolo_max": float(conf.max()) if len(conf) else 0.0,
         "n_boxes_op": int((conf >= ymeta["conf"]).sum())}
    for k, cname in yolo.names.items():
        d[f"max_{cname}"] = float(conf[cls == k].max()) if (cls == k).any() else 0.0
    rows.append(d)
    if i % 500 == 0: print(i, len(m), flush=True)
s = pd.DataFrame(rows)
out = m.merge(s, on="image")
out.to_csv(OUT, index=False)
n = len(m) - 20
print(f"timing ms/img classifier={tc/n*1000:.2f} yolo={ty/n*1000:.2f}")
