import random, sys
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO
REPO = Path(__file__).resolve().parents[5]
OUT = Path(__file__).parent

def panel(model, img, label=None):
    im = Image.open(img).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im)
    if label and label.exists():
        for l in label.read_text().splitlines():
            _, x, y, w, h = map(float, l.split()[:5]); d.rectangle(((x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H), outline="lime", width=4)
    for b in model.predict(img, conf=0.25, verbose=False)[0].boxes:
        x0, y0, x1, y1 = b.xyxy[0].tolist(); d.rectangle((x0, y0, x1, y1), outline="red", width=3); d.text((x0+2, y0+2), f"{float(b.conf):.2f}", fill="red")
    im.thumbnail((400, 400)); return im

if __name__ == "__main__":
    random.seed(2); m = YOLO(REPO / "models/yolo_caries/weights/feasibility_ai_best.pt")
    lab = OUT / "external/od_caries/labels/test"; img = OUT / "external/od_caries/images/test"
    od = random.sample([p for p in lab.iterdir() if p.read_text().strip()], 8)
    kn = random.sample(sorted((REPO / "data/raw/kaggle_oral/Normal").iterdir()), 8)
    for tag, items in (("od", [(img / f"{p.stem}.jpg", p) for p in od]), ("kaggle_normal", [(p, None) for p in kn])):
        s = Image.new("RGB", (1600, 800), "white")
        for i, (p, l) in enumerate(items): s.paste(panel(m, p, l), ((i % 4) * 400, (i // 4) * 400))
        s.save(OUT / f"external_{tag}.jpg")
