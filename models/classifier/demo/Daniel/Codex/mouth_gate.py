"""Mouth gate: decide what an uploaded photo shows before it reaches the classifier.

The classifier was trained only on intraoral close-ups (teeth / gums / tongue filling
the frame), so a selfie, a distant portrait or an unrelated picture would get a
confident-looking but meaningless score. The gate routes each upload one of three ways:

  intraoral  — already a close-up of the mouth/teeth: passed through unchanged.
  face_crop  — a face photo: the mouth is located with OpenCV's YuNet face detector
               (5 landmarks incl. both mouth corners), the image is de-rotated so the
               lips are level, and a 4:3 box around the mouth is cropped for the model.
               The crop must show an open mouth (CLIP check), else the photo is rejected.
  rejected   — no usable mouth (not a mouth photo at all, mouth too small / closed,
               or a face whose mouth could not be located). The classifier is not run.

Scene understanding is CLIP zero-shot (openai/clip-vit-base-patch16 via transformers):
the image is scored against prompt groups ("intraoral", "face", "other") and each
group's probability is the sum over its prompts. No training data needed.

Both models download on first use (CLIP ~600 MB to the Hugging Face cache, YuNet
~230 KB next to this file). If either cannot be loaded the gate reports itself as
unavailable and the demo falls back to classifying the whole image.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
CLIP_NAME = "openai/clip-vit-base-patch16"

# Each group's probability is the summed softmax over its prompts.
SCENE_PROMPTS = {
    "intraoral": [
        "a close-up photo of teeth",
        "an intraoral dental photo of teeth and gums",
        "a close-up photo of the inside of a mouth",
        "a close-up photo of an open mouth showing teeth",
        "a close-up photo of a tongue",
        "a close-up photo of lips and teeth",
        "a dental photo of decayed teeth",
        "a close-up photo of a mouth ulcer",
    ],
    "face": [
        "a photo of a person's face",
        "a portrait photo of a person",
        "a selfie of a smiling person",
        "a photo of people",
        "a photo of a person standing",
    ],
    "other": [
        "a photo of an animal",
        "a photo of food",
        "a photo of a landscape",
        "a photo of a building",
        "a photo of an object",
        "a photo of a book",
        "a screenshot of text",
        "a document",
        "a photo of a room",
        "a photo of a hand",
        "a dental x-ray",
        "a drawing or cartoon",
        "a blank or blurry image",
    ],
}
OPEN_PROMPTS = {
    "open": [
        "a close-up photo of an open mouth showing teeth",
        "a close-up photo of a smile showing teeth",
        "a close-up photo of teeth",
    ],
    "closed": [
        "a close-up photo of closed lips",
        "a close-up photo of a closed mouth with no teeth visible",
        "a close-up photo of a mustache and closed lips",
    ],
}


@dataclass
class GateConfig:
    face_score: float = 0.80       # YuNet confidence that accepts a face on its own
    face_score_clip: float = 0.50  # ...or this much if CLIP also says "face" (p_face >= clip_face)
    clip_face: float = 0.50
    det_sides: tuple = (640, 320)  # YuNet's score is scale-sensitive: detect at both, keep the best
    face_mouth_frac: float = 0.40  # mouth-corner distance / image width below this => "face photo, crop"
    min_mouth_px: int = 40         # mouth-corner distance (px in the <=1280 px working image) => too small
    crop_w: float = 1.7            # crop width  in mouth-corner distances
    crop_h: float = 1.3            # crop height in mouth-corner distances (≈4:3 like the training photos)
    crop_dy: float = 0.15          # shift the crop centre down (open mouths drop the lower lip)
    reject_other: float = 0.55     # p(other) at/above this with no face => not a mouth photo
    min_open: float = 0.35         # p(open) on the crop below this => mouth closed, reject
    # Intraoral close-ups occasionally fire a phantom "face" (molars as eyes). Ignore detections
    # when CLIP is sure the photo is a mouth close-up and not a face.
    ignore_face_if_intraoral: float = 0.50
    ignore_face_if_face_below: float = 0.10


@dataclass
class GateResult:
    route: str                              # "intraoral" | "face_crop" | "rejected"
    image: Image.Image | None               # what the classifier should see (None when rejected)
    reason: str = ""                        # short machine code for the outcome
    message: str = ""                       # human-readable explanation
    scene: dict = field(default_factory=dict)
    box: list | None = None                 # crop corners in source-image fractions, [[x,y]*4]
    faces: int = 0
    p_open: float | None = None

    def as_dict(self) -> dict:
        return {"route": self.route, "reason": self.reason, "message": self.message,
                "scene": {k: round(v, 4) for k, v in self.scene.items()},
                "box": self.box, "faces": self.faces,
                "p_open": None if self.p_open is None else round(self.p_open, 4)}


class MouthGate:
    def __init__(self, device: torch.device, asset_dir: Path, cfg: GateConfig | None = None):
        import cv2  # local import: only the gate needs OpenCV
        from transformers import CLIPModel, CLIPProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # silence the CLIP load report on startup

        self.cv2 = cv2
        self.cfg = cfg or GateConfig()
        self.device = device

        onnx = Path(asset_dir) / "face_detection_yunet_2023mar.onnx"
        if not onnx.exists():
            onnx.parent.mkdir(parents=True, exist_ok=True)
            tmp = onnx.with_suffix(".part")
            urllib.request.urlretrieve(YUNET_URL, tmp)
            tmp.replace(onnx)
        self.detector = cv2.FaceDetectorYN.create(str(onnx), "", (320, 320),
                                                  score_threshold=self.cfg.face_score_clip)

        self.processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        self.clip = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
        if device.type == "cuda":
            self.clip.half()
        self._scene = self._encode_prompts(SCENE_PROMPTS)
        self._open = self._encode_prompts(OPEN_PROMPTS)

    # ---- CLIP -------------------------------------------------------------

    @torch.no_grad()
    def _encode_prompts(self, groups: dict[str, list[str]]):
        names, texts = [], []
        for g, ps in groups.items():
            names += [g] * len(ps)
            texts += ps
        tok = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        emb = self.clip.get_text_features(**tok)
        emb = getattr(emb, "pooler_output", emb)  # transformers 5 may return a model output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return names, emb

    @torch.no_grad()
    def _image_emb(self, im: Image.Image) -> torch.Tensor:
        px = self.processor(images=im, return_tensors="pt")["pixel_values"].to(self.device, self.clip.dtype)
        emb = self.clip.get_image_features(pixel_values=px)
        emb = getattr(emb, "pooler_output", emb)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _group_probs(self, img_emb: torch.Tensor, prompts) -> dict[str, float]:
        names, txt = prompts
        logits = self.clip.logit_scale.exp().float() * (img_emb.float() @ txt.float().T)[0]
        p = logits.softmax(0).cpu().numpy()
        out: dict[str, float] = {}
        for n, v in zip(names, p):
            out[n] = out.get(n, 0.0) + float(v)
        return out

    # ---- faces ------------------------------------------------------------

    def _faces(self, im: Image.Image) -> np.ndarray:
        """YuNet rows (in `im` pixels): x, y, w, h, eyeR, eyeL, nose, mouthR, mouthL (x,y pairs), score.

        Runs at each of cfg.det_sides and merges with NMS, since a face can score 0.4 at one
        scale and 0.9 at another."""
        rows = []
        for side in self.cfg.det_sides:
            s = min(1.0, side / max(im.size))
            small = im if s == 1.0 else im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                                                  Image.BILINEAR)
            bgr = np.ascontiguousarray(np.asarray(small)[:, :, ::-1])
            self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = self.detector.detect(bgr)
            if faces is not None:
                faces = faces.copy()
                faces[:, :14] /= s
                rows.append(faces)
        if not rows:
            return np.zeros((0, 15), np.float32)
        faces = np.concatenate(rows)
        keep = self.cv2.dnn.NMSBoxes(faces[:, :4].tolist(), faces[:, 14].tolist(), 0.0, 0.3)
        return faces[np.asarray(keep, int).reshape(-1)]

    def _mouth_crop(self, im: Image.Image, f: np.ndarray):
        c = self.cfg
        mr, ml = f[10:12].astype(float), f[12:14].astype(float)
        d = float(np.linalg.norm(ml - mr))
        cx, cy = (mr + ml) / 2
        ang = math.atan2(ml[1] - mr[1], ml[0] - mr[0])  # roll of the lip line (radians)
        # Box in the de-rotated frame, centred a bit below the corner midpoint.
        w, h = c.crop_w * d, c.crop_h * d
        ccx, ccy = cx - math.sin(ang) * c.crop_dy * d, cy + math.cos(ang) * c.crop_dy * d
        rot = im.rotate(math.degrees(ang), resample=Image.BICUBIC, center=(ccx, ccy), fillcolor=(0, 0, 0))
        crop = rot.crop((round(ccx - w / 2), round(ccy - h / 2), round(ccx + w / 2), round(ccy + h / 2)))
        # Corners of that box mapped back to the source image, as fractions (for the UI overlay).
        ca, sa = math.cos(ang), math.sin(ang)
        pts = []
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
            pts.append([round((ccx + dx * ca - dy * sa) / im.width, 4),
                        round((ccy + dx * sa + dy * ca) / im.height, 4)])
        return crop, pts, d

    # ---- main entry -------------------------------------------------------

    def __call__(self, im: Image.Image) -> GateResult:
        c = self.cfg
        scene = self._group_probs(self._image_emb(im), self._scene)
        faces = self._faces(im)
        need = c.face_score_clip if scene["face"] >= c.clip_face else c.face_score
        faces = faces[faces[:, 14] >= need]
        if scene["intraoral"] >= c.ignore_face_if_intraoral and scene["face"] < c.ignore_face_if_face_below:
            faces = faces[:0]  # phantom face on a close-up
        n = len(faces)

        if n:
            # Largest face wins (selfie subject); YuNet already dropped low-score boxes.
            f = faces[np.argmax(faces[:, 2] * faces[:, 3])]
            d = float(np.linalg.norm(f[12:14] - f[10:12]))
            if d / im.width < c.face_mouth_frac:
                if d < c.min_mouth_px:
                    return GateResult("rejected", None, "mouth_too_small",
                                      "Found a face, but the mouth is too small in the frame to assess. "
                                      "Take a closer photo of the open mouth.", scene, faces=n)
                crop, box, _ = self._mouth_crop(im, f)
                p_open = self._group_probs(self._image_emb(crop), self._open)["open"]
                if p_open < c.min_open:
                    return GateResult("rejected", None, "mouth_closed",
                                      "Found the mouth, but it looks closed — open wide or smile so the "
                                      "teeth and gums are visible, then retake.", scene, box, n, p_open)
                msg = "Face photo: cropped to the mouth before classifying."
                if n > 1:
                    msg += f" {n} faces found; used the largest."
                return GateResult("face_crop", crop, "face_crop", msg, scene, box, n, p_open)

        top = max(scene, key=scene.get)
        if scene["other"] >= c.reject_other or (top == "other" and scene["intraoral"] < 0.25):
            return GateResult("rejected", None, "no_mouth",
                              "This doesn't look like a photo of a mouth or teeth, so it was not classified.",
                              scene, faces=n)
        if top == "face" and scene["intraoral"] < 0.25:
            return GateResult("rejected", None, "mouth_not_found",
                              "This looks like a photo of a person, but the mouth couldn't be located. "
                              "Take a front-facing photo with the mouth open, or a close-up of the teeth.",
                              scene, faces=n)
        return GateResult("intraoral", im, "intraoral", "Close-up of the mouth: classified as-is.", scene, faces=n)
