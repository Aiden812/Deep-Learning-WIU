# Daniel's work: what was built, why, and what it showed

**IT2226 AI Innovator and Deep Learning Project: Dental Health Awareness and Affordability in Singapore**
Covers 2026-11-05 to 2026-11-16.

This document summarises everything in the `Daniel` folders: the data pipeline, the two models, the demo and the prototype. For each piece it gives the reason behind the main decisions and the measured results. The detailed technical report is [report_classifier_yolo_qwen.md](report_classifier_yolo_qwen.md); the notebooks hold the full analysis.

---

## Contents

1. [The system in one page](#1-the-system-in-one-page)
2. [Timeline](#2-timeline)
3. [Phase 1: a 6-class classifier on the Kaggle data](#3-phase-1-a-6-class-classifier-on-the-kaggle-data-2026-11-05)
4. [Phase 2: building a bigger dataset with Qwen](#4-phase-2-building-a-bigger-dataset-with-qwen)
5. [Stage 1: the binary "issue or not" classifier](#5-stage-1-the-binary-issue-or-not-classifier)
6. [Stage 2: the YOLO detector](#6-stage-2-the-yolo-detector)
7. [Classifier vs YOLO: is stage 1 still needed?](#7-classifier-vs-yolo-is-stage-1-still-needed)
8. [Demo app and SmileCheck SG prototype](#8-demo-app-and-smilecheck-sg-prototype)
9. [Engineering decisions](#9-engineering-decisions)
10. [Limitations and responsible AI](#10-limitations-and-responsible-ai)
11. [Lessons learned](#11-lessons-learned)
12. [Open items](#12-open-items)
13. [Where things are](#13-where-things-are)

---

## 1. The system in one page

```
 photo / video frame
        │
        ▼
 ┌─────────────────┐   not a mouth, mouth closed or too small → "please retake"
 │   Mouth gate    │   face photo → crop to the mouth
 │ (CLIP + YuNet)  │   intraoral close-up → use as it is
 └────────┬────────┘
          ▼
 ┌─────────────────┐   p(issue) < 0.729 → "none of the five screened conditions seen"
 │ Stage 1         │
 │ MobileNetV3-L   │   p(issue) ≥ 0.729 → flagged, recommend a check-up
 │ "issue or not?" │
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ Stage 2         │   boxes + names: Calculus, Caries, Hypodontia,
 │ YOLO11-medium   │   Mouth_Ulcer, Tooth_Discoloration
 │ "where, what?"  │
 └─────────────────┘
```

| Part | What it is | Headline result |
|---|---|---|
| Dataset | 33,952 images from 6 sources, cleaned and labelled with the help of a local Qwen vision-language model | Val/test condition labels are all drawn by people |
| Stage 1 classifier | MobileNetV3-Large, 4.2 M parameters, 8 ms per image on CPU | Test macro-F1 0.975; **0.944 on photos from other sources** (`xsrc_test`) |
| Stage 2 detector | YOLO11-medium (retrained 2026-11-16) | Test mAP50 0.821, mAP50-95 0.435; boxes 98.9% of issue photos, 2.6% false alarms on healthy photos |
| Demo | FastAPI web app with a mouth gate, the classifier and the detector | Runs both models, explains each result, keeps no photos |
| Prototype | SmileCheck SG, a patient-facing page on top of the demo | Photos and videos, plain-language explanations, clinic map |

**The most important finding.** In val/test, every photo with a condition comes from one source (`roboflow_oral`, 640×640 exports) and every healthy photo from other sources. A model can score well by recognising Roboflow photos rather than disease. So every number is reported twice: on the normal test set (an upper bound) and on a **cross-source** set of photos from other sources that contain both labels.

---

## 2. Timeline

| Date | Work | Commit |
|---|---|---|
| 2026-11-05 | 6-class classifier on Kaggle oral images, 13 backbones compared; first web demo | `2562f72`, `10f23d7` |
| 2026-11-05 to 11-10 | Web image scraper (Baidu); Qwen cleaning and labelling pipeline; box quality check against human boxes | |
| 2026-11-11 | Qwen labels for the full dataset; label fixes 7a (images nobody checked are no longer "Normal"); binary classifier v1 and v2 | `3016a18`, `0e42971`, `17071f7` |
| 2026-11-12 | Mouth gate in the demo; first YOLO sweep (nano, small, medium, person-only ablation) | `4b20f90` |
| 2026-11-13 | Hypodontia re-check 7b and gap boxes 7c; stable split; classifier v3 | |
| 2026-11-14 | YOLO11-small retrained on the new data; detector added to the demo; prototype updated | `b849ee7`, `f186240` |
| 2026-11-16 | All YOLO sizes retrained, YOLO11-medium selected; classifier vs YOLO comparison; demo option to switch models off | `c6b6615`, `e5f4ecd` |

---

## 3. Phase 1: a 6-class classifier on the Kaggle data (2026-11-05)

**Goal.** Take an intraoral photo and return probabilities for six classes: calculus, caries, hypodontia, mouth ulcer, normal and tooth discoloration. Data: the brief's Kaggle oral dataset only (4,375 photos).

### What was found in the data

| Finding | Why it matters | What was done |
|---|---|---|
| **The dataset was augmented before it was published.** About 875 discoloration files follow Keras `ImageDataGenerator` naming; calculus and hypodontia hold rotated, sheared and recoloured copies | A random split would put copies of a training photo into about half of the test images, so the test score would be inflated | Duplicates grouped by MD5, perceptual hash (≤ 6 bits) and CNN-embedding similarity (≥ 0.90); train/val/test split **by photo group**; val/test keep one image per group |
| **The imbalance in the brief is wrong.** Calculus has 1,296 files but only about 255 distinct photos | Balancing by file count would over-weight a few calculus photos | The sampler balances by class *and* by photo group |
| **Normal photos come from a different camera** (phone photos, about 10× the resolution) | The model can learn "high resolution = healthy" | All images cached at ≤ 384 px; blur and JPEG augmentation; low-resolution robustness test |
| **The same photo is filed under two classes** (e.g. calculus and hypodontia) | One label per image is wrong for mouths with several conditions | Such groups go to training only; the app shows the top two classes |

### Results

13 ImageNet-pretrained backbones were fine-tuned with one recipe (B: balanced sampler, strong augmentation, label smoothing) plus a naive baseline (A: plain shuffle, flips only). The selection rule was fixed before training: mean macro-F1 on clean val and a "phone-quality" copy of val; runs within 0.01 are tied, and the fastest on CPU wins.

| Model | Params (M) | CPU ms | Test macro-F1 [95% CI] |
|---|---:|---:|---|
| **B EfficientNet-B0 (selected)** | 4.0 | 9.4 | **0.936 [0.900, 0.965]** |
| B ConvNeXt-Base | 87.6 | 70.3 | 0.945 [0.912, 0.972] |
| B ConvNeXt-Small | 49.5 | 44.4 | 0.935 [0.901, 0.965] |
| B Swin-T | 27.5 | 27.5 | 0.928 [0.890, 0.960] |
| B ConvNeXt-Tiny | 27.8 | 23.6 | 0.925 [0.885, 0.958] |
| B MobileNetV3-Large | 4.2 | 7.2 | 0.905 [0.862, 0.939] |
| B ViT-B/16 | 85.8 | 76.8 | 0.900 [0.855, 0.939] |
| A ConvNeXt-Tiny (baseline) | 27.8 | 23.5 | 0.915 [0.876, 0.947] |

- The test set had only 321 unique photos, so the confidence intervals overlap for almost every model. A model 20× larger was not measurably better.
- **Caries was the weak class** (recall 0.80), and the score dropped to 0.836 on low-resolution, compressed photos.

### Why this was not the final design

- **Too little data.** About 1,000 distinct photos after de-duplication, with only 25–40 test photos in the small classes.
- **One label per photo** does not fit mouths that show several conditions, and it cannot show *where* the problem is.
- **The Normal class is confounded with its camera.**

This led to the **two-stage design**: a binary classifier that only decides "is there an issue?" and a YOLO detector that finds and names each condition. Both are trained on a much larger dataset that combines several sources (phase 2).

---

## 4. Phase 2: building a bigger dataset with Qwen

Notebook: `models/classifier/notebooks/Daniel/qwen_clean_and_label.ipynb` (terminal runner: `scripts/Daniel/run_qwen_pipeline.py`).

### 4.1 Why Qwen

The available datasets each carry partial labels. Roboflow has human boxes for every condition; dentalai and zenodo only mark caries; kaggle only gives a folder name; mendeley says "no caries"; web images only have a search term. Labelling tens of thousands of photos by hand was not possible in the time. A local vision-language model (`qwen3.8-flash-next-radiance`, served by llama-swap, thinking off) filled the gaps. It was set up to be **conservative**, and **val/test never depend on it**.

### 4.2 Sources and how each one was handled

| Source | What it gives | Route |
|---|---|---|
| `roboflow_oral` | Human boxes for all five conditions | `gt`: no Qwen request |
| `dentalai`, `zenodo` | Human caries boxes only | `gt_check`: Qwen does a light check for the other four conditions |
| `kaggle_oral` | Condition per folder | `known`: Qwen must box the named condition |
| `mendeley_noncaries` | Caries-free close-ups | `label`: Qwen labels and boxes; caries is not offered |
| Web (`data/raw/external`) | 21,000 scraped images | `web`: Qwen screens, labels and boxes |

**Web scraping.** `scrape_web_images.py` searches **Baidu with Chinese clinical terms** (e.g. 龋齿 口内照). English queries mostly returned book covers, Google blocks scripts, and Bing returned unrelated images to scripted requests. Perceptual-hash de-duplication brought 21,000 images down to 12,968 unique ones (3,843 were found under more than one class). Qwen's screen accepted only **665 (5%)**: real teeth, unobstructed, close-up, a photograph, not a collage.

### 4.3 Keeping Qwen honest

| Measure | Why |
|---|---|
| Prompt says "a false alarm is worse than a miss"; evidence first, then label, then confidence; only `high` findings kept | Stops guessing |
| A look-alike guide: calculus = material added, caries = material lost (a hole), discoloration = colour only on a smooth surface; plus a list of non-conditions (shadows, fillings, slightly yellow teeth) | These three were often confused |
| Every answer cached as JSONL per step and model | Runs can be interrupted and resumed; thresholds can change without new requests |

Measured effect of the new label prompt (130 kaggle/mendeley images):

| | Old prompt | New prompt |
|---|---:|---:|
| kaggle `Normal` labelled Normal | 12% | **70%** |
| Caries found on kaggle `Normal` | 48% | **3%** |
| Caries found on caries-free mendeley | 37% | **8%** |
| Caries found on kaggle discoloration | 49% | **10%** |

**Qwen's boxes are rough.** Against 32 human-boxed images, mean IoU was only 0.19 (precision 0.31). The 0–1000 coordinate reading was confirmed correct (centre offset within ±0.02), so the error is Qwen's localisation. This is why **val/test use only human boxes**.

### 4.4 Filtering

Boxes were checked by eye, and the main problems became rules:

| Dropped | Why | Count |
|---|---|---:|
| Qwen hypodontia boxes (except kaggle's Hypodontia folder) | On side views Qwen marked hidden or out-of-frame teeth as missing (39% of upper-left/right views vs 4% of lower-front) | 258 boxes |
| Every mendeley image Qwen flagged | Its calculus and hypodontia findings were unreliable | 4,515 images |
| Qwen boxes covering > 60% of the image | When told a condition must be boxed, Qwen sometimes boxed the whole photo | 54 boxes |
| Light-check boxes wrapping a human caries box | Qwen re-boxed marked caries as discoloration | 83 boxes |
| Human boxes thinner than 0.5% of the image | About 3 px, too small to learn | 588 boxes |
| kaggle copies of roboflow photos | Same photo; keep the copy with human boxes | 2,773 images |

### 4.5 Label fixes (7a): "Normal" must mean someone checked

The first classifier's most confident "errors" on Normal images showed real decay and staining. Two rules had made photos "Normal" without anyone looking:

| Problem | Fix |
|---|---|
| zenodo annotations cover only about a third of its photos, and "no annotation file" had been read as "no caries", so about 3,400 never-checked photos became Normal | Only annotated zenodo photos count (4,039 dropped) |
| A web image found by a condition search became Normal when Qwen found nothing | It stays Normal only if every search that found it was a Normal search (75 dropped) |

**This was the single biggest improvement in the project** (section 5.3).

### 4.6 Missing teeth outside Roboflow (7b and 7c)

**Why.** The demo called a photo with an obvious missing front tooth "normal". Nearly all hypodontia examples came from Roboflow, whose photos are heavily processed: a gap there is a solid black hole. Qwen's other hypodontia findings had been dropped with their bad boxes, so clinical close-ups with gaps were all labelled healthy. The classifier flagged **0.1%** of mendeley photos where Qwen had seen a gap, against 97% of Roboflow's.

**7b: one stricter question.** "Is a tooth missing?" A tooth counts only if there is a tooth-sized gap inside the row that nothing else explains (hidden tooth, angle, photo edge, narrow space, a child's baby tooth, a false tooth). Asked of 2,013 images; **143 confirmed**. 109 of them had no other boxes and became **classifier-only** issue images.

**7c: gap boxes, checked by a second request instead of a person.** Qwen draws a box using its own 7b note on where the gap is. The photo is then sent again with the box drawn in green, and Qwen is asked what is inside it.

| | Count |
|---|---:|
| Boxes drawn | 144 |
| Pass shape rules (≤ 15% of image, not a sliver, ≤ 6 per photo) | 132 |
| Rejected by the check (16 on a tooth, 9 on a broken tooth) | 25 |
| **Kept** | **107 on 106 photos** |

On inspection, about **8 in 10** kept boxes sit on the gap (up from about 6 in 10 for the original boxes). 63 classifier-only photos joined YOLO's train set, 26 YOLO photos got their gap labelled, and **17 were held out** (seeded 20%) to test YOLO on missing teeth outside Roboflow.

### 4.7 The split

- **Copies stay together.** Roboflow augmentations, perceptual-hash near-duplicates and kaggle re-uploads of Roboflow photos are grouped and always share a split.
- **Val/test hold only photos whose condition labels were drawn by people.** Issue photos are Roboflow groups; healthy photos are a share of every source's Normal groups. `data_person.yaml` gives the same exam with only fully person-labelled images.
- **Stable split (2026-11-13).** Val/test used to be drawn by shuffling group numbers, which shift whenever an image is added. Adding the 109 missing-tooth photos would have reshuffled about 60% of the healthy val/test images. Groups are now ordered by a seeded hash of their smallest MD5. Adding the 109 images then moved 0 val and 2 test images. The switch redrew val/test once, so **models trained before 2026-11-13 must not be scored on the current val/test**.

### 4.8 Final dataset (build v4, 2026-11-13)

| | train | val | test |
|---|---:|---:|---:|
| Calculus | 3,661 | 369 | 401 |
| Caries | 10,738 | 830 | 837 |
| Hypodontia | 2,628 | 311 | 335 |
| Mouth ulcer | 4,699 | 574 | 543 |
| Tooth discoloration | 7,832 | 937 | 898 |
| Healthy, in YOLO (capped at 10% of train) | 2,603 | 609 | 621 |
| **YOLO images** | **26,032** | **2,941** | **2,936** |
| Classifier: issue / healthy | 23,475 / 4,600 | 2,332 / 609 | 2,315 / 621 |

33,952 images in total (3.64 GB zip). **Healthy photos are capped at 10% of YOLO's train set** because a detector that sees too many empty images learns to stay quiet; the classifier gets all of them.

| Build | Date | Change | Trained on it |
|---|---|---|---|
| v1 | 2026-11-11 | First full Qwen build | classifier v1 |
| v2 | 2026-11-11 | + 7a label fixes | classifier v2, YOLO sweep 2026-11-12 |
| v3 | 2026-11-13 | + 7b missing teeth (classifier only), stable split | classifier v3 (current) |
| v4 | 2026-11-13 | + 7c gap boxes (YOLO only) | YOLO 2026-11-14 and 2026-11-16 (current) |

---

## 5. Stage 1: the binary "issue or not" classifier

Notebook: `models/classifier/notebooks/Daniel/oral_condition_classifier.ipynb`.

### 5.1 Design decisions

| Decision | Why |
|---|---|
| **Binary** (issue vs normal), not 6 classes | Naming and locating conditions is YOLO's job; one yes/no question is easier to learn and to calibrate |
| **Cross-source check sets** `xsrc_val` / `xsrc_test` (about 1,180 images each), carved by photo group from non-Roboflow training sources, which contain both labels | Val/test cannot tell "detects disease" from "detects Roboflow"; these sets can |
| **Balanced sampler**: healthy and issue drawn equally; within issue, Roboflow and other sources half each; every photo group equally likely | So the model cannot learn "not Roboflow means healthy" from how often it sees each source |
| **Strong augmentation** incl. blur and JPEG; tiny hue jitter | Patients upload poor phone photos; colour is diagnostic for discoloration and calculus |
| **224×224 squash**, no centre crop | Lip ulcers near the edge are not cut off |
| **Label smoothing 0.1** | Some labels are Qwen's; the model should not be pushed to 100% confidence |
| **Selection rule fixed before training**: mean macro-F1 of val, phone-quality val and `xsrc_val`; within 0.01 tied; fastest on CPU wins | `xsrc_val` stops the rule rewarding the shortcut; CPU speed matters for the app server |
| **Threshold for 95% sensitivity on both val and `xsrc_val`** | A "normal" result skips YOLO, so a missed condition is silent; a false flag only costs a check-up |
| **Temperature scaling** | "70%" should mean about 70% |

### 5.2 Results (v3, current)

| Run | Params (M) | CPU ms | test macro-F1 | `xsrc_test` macro-F1 [95% CI] | `xsrc_test` AUROC |
|---|---:|---:|---:|---|---:|
| B EfficientNet-V2-S | 20.2 | 25.9 | 0.976 | 0.947 [0.934, 0.961] | 0.977 |
| B ResNet-50 | 23.5 | 21.6 | 0.975 | 0.946 [0.932, 0.959] | 0.976 |
| B EfficientNet-B0 | 4.0 | 10.9 | 0.976 | 0.945 [0.930, 0.959] | 0.975 |
| **B MobileNetV3-Large (selected)** | **4.2** | **8.3** | **0.975** | **0.944 [0.929, 0.957]** | **0.978** |
| B ConvNeXt-Tiny | 27.8 | 30.4 | 0.969 | 0.948 [0.935, 0.960] | 0.976 |
| B Swin-T | 27.5 | 31.1 | 0.974 | 0.947 [0.932, 0.960] | 0.980 |
| B DenseNet-121 | 7.0 | 23.5 | 0.971 | 0.947 [0.934, 0.960] | 0.968 |
| A ConvNeXt-Tiny (baseline) | 27.8 | 28.4 | 0.974 | 0.939 [0.925, 0.952] | 0.986 |

- **Architecture and size made no measurable difference.** All B runs overlap, so the 4 M-parameter MobileNet won on speed.
- **Recipe B's real gain is robustness.** Under blur, the baseline's `xsrc_test` macro-F1 fell to 0.854 (sensitivity 0.751), while B stayed at 0.943.

| Degradation (`xsrc_test` macro-F1) | A baseline | B ConvNeXt-Tiny | B MobileNetV3 |
|---|---:|---:|---:|
| original | 0.939 | 0.948 | 0.944 |
| low-res 160 px + JPEG 60 | 0.897 | 0.933 | 0.928 |
| blur (r = 2) | **0.854** | 0.943 | 0.943 |
| JPEG q = 20 | 0.897 | 0.944 | 0.947 |
| bright ×1.6 | 0.911 | 0.924 | 0.929 |

**At the operating point** (temperature 0.706, threshold 0.729):

| Set | Sensitivity | Specificity | Macro-F1 | ECE raw → calibrated |
|---|---:|---:|---:|---|
| test | 0.992 | 0.947 | 0.973 | 0.043 → 0.010 |
| test, person-labelled | 0.992 | 0.881 | 0.897 | 0.050 → 0.018 |
| `xsrc_test` | 0.928 | 0.963 | 0.945 | 0.036 → 0.028 |

### 5.3 How the classifier improved: v1 → v2 → v3

| Version | Data | Selected | `xsrc_test` macro-F1 (thr 0.5) | `shortcut_gap` (val − `xsrc_val`) | Threshold for 95% sens. | `xsrc_test` spec. at threshold |
|---|---|---|---:|---:|---:|---:|
| v1 | before 7a | ResNet-50 | 0.884 | ~0.10 | 0.106 | 0.733 |
| v2 | 7a label fixes | MobileNetV3 | 0.950 | ~0.02 | 0.847 | 0.958 |
| v3 | + 7b missing teeth, stable split | MobileNetV3 | 0.944 | ~0.03 | 0.729 | 0.963 |

- **The label fix (7a) mattered more than any model choice.** Unchecked "Normal" photos that actually showed decay taught the model that decay can be normal. Removing them shrank the gap between the val score and the cross-source score from about 0.10 to about 0.02. To keep 95% sensitivity, v1 needed a threshold of 0.106, which flagged 27% of healthy cross-source photos; v2 flagged 4%. The test sets also changed with the fix, so read this as a large improvement rather than an exact number.
- **v2 vs v3, fairly compared** on the 1,031 held-out photos neither model trained on: v3 is slightly better (sensitivity 0.971 → 0.975, specificity 0.936 → 0.947, AUROC 0.976 → 0.979).

### 5.4 Weak spot: missing teeth in clinical close-ups

| Condition (other sources, held out) | Images | v2 | v3 |
|---|---:|---:|---:|
| Caries | 152 | 0.967 | 0.980 |
| Tooth discoloration | 21 | 0.905 | 0.905 |
| Mouth ulcer | 9 | 1.000 | 1.000 |
| Hypodontia, Roboflow | 40 | 0.975 | 1.000 |
| **Hypodontia, other sources** | **13** | **0.308** | **0.385** |

v3 now *ranks* mendeley's missing-tooth photos above its healthy ones far more often (AUROC on mendeley 0.69 → 0.94), but most still sit below the threshold (P(issue) ≈ 0.02). Only about 87 of the new photos are in training, among 23,000 issue images, and the sampler does not balance by condition, so each epoch draws only about 20 of them. **Next step:** over-sample non-Roboflow issue photos and retrain MobileNet only (about 6 minutes).

---

## 6. Stage 2: the YOLO detector

Notebook: `models/yolo_caries/notebooks/Daniel/yolo_oral_detector.ipynb`.

### 6.1 Design decisions

| Decision | Why |
|---|---|
| COCO-pretrained YOLO11 nano / small / medium, one recipe | Size is the only difference between runs |
| Ablation `yolo11s_person`: trains only on images labelled entirely by people | Measures whether Qwen's labels help or hurt |
| Hue jitter almost off, saturation/brightness jitter reduced | A hue shift can turn a stain into a healthy shade |
| No vertical flip, no rotation | Phone photos of a mouth are upright and close to level |
| One-time resized copy of images (≤ 1,024 px) | zenodo photos reach ~8,700 px; with mosaic, decoding them left the GPU waiting |
| Selection: highest val mAP50-95; within 0.005 tied; fewest GFLOPs wins | The model runs behind a web app |
| Operating confidence = max mean F1 on val | |
| Test scored twice: all images and person-labelled only | Checks whether Qwen's "Normal" labels distort the score |
| Image-level scores (boxed issue photos, false alarms on healthy photos) | mAP says little about what a user sees |

### 6.2 Model history

| Date | Data | Chosen | Why | Test mAP50 / mAP50-95 |
|---|---|---|---|---|
| 2026-11-12 | v2 | yolo11s | yolo11m (0.446) was within 0.005 of yolo11s (0.444) at 3× the compute | 0.450 mAP50-95 on v2's test (not comparable, split redrawn) |
| 2026-11-14 | v4 | yolo11s | Only yolo11s retrained, to add the 7c gap boxes | 0.814 / 0.430 |
| **2026-11-16** | **v4** | **yolo11m** | All four retrained; yolo11m beat yolo11s by 0.007, beyond the 0.005 tie margin | **0.821 / 0.435** |

**Fair check of the 11-14 retrain.** On 708 test photos that neither the 11-12 nor the 11-14 model trained on, they tie (mAP50-95 0.397 vs 0.401). The 11-14 model is a little better on hypodontia (AP50 0.539 → 0.565).

### 6.3 Current sweep (2026-11-16, build v4)

| Run | Params (M) | GFLOPs | min / epoch | Epochs (best) | GPU ms | val mAP50 | val mAP50-95 | val person mAP50-95 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| yolo11n | 2.6 | 6.6 | 1.5 | 80 (80) | 1.0 | 0.807 | 0.434 | 0.435 |
| yolo11s | 9.4 | 21.7 | 2.2 | 78 (62) | 2.0 | 0.825 | 0.445 | 0.445 |
| **yolo11m (chosen)** | **20.1** | **68.3** | **4.4** | **69 (59)** | **5.2** | **0.832** | **0.452** | **0.452** |
| yolo11s_person | 9.4 | 21.7 | 1.6 | 57 (42) | 2.3 | 0.795 | 0.429 | 0.442 |

yolo11n was still improving at epoch 80, so it might gain from a longer run.

### 6.4 Test results (yolo11m, operating confidence 0.361)

| | All test (2,936) | Person-labelled (2,374) |
|---|---:|---:|
| Precision / recall | 0.807 / 0.768 | 0.807 / 0.768 |
| mAP50 | 0.821 | 0.822 |
| mAP50-95 | 0.435 | 0.435 |

| Condition | AP50 | AP50-95 | Image recall | Image precision |
|---|---:|---:|---:|---:|
| Calculus | 0.643 | 0.302 | 0.820 | 0.846 |
| Caries | 0.818 | 0.441 | 0.875 | 0.947 |
| Hypodontia | 0.836 | 0.365 | 0.934 | 0.937 |
| Mouth ulcer | 0.918 | 0.483 | 0.978 | 0.983 |
| Tooth discoloration | 0.890 | 0.583 | 0.904 | 0.871 |

**Image level:** 98.9% of the 2,315 issue photos get at least one box. 2.6% of the 621 healthy photos get a false alarm. On the 59 person-labelled healthy photos (kaggle's Normal folder) the rate is 11.9% (7 photos); the 2026-11-14 yolo11s had 1.7% there, so these photos should be checked.

**Findings:**
- **Qwen's "Normal" labels do not distort the score.** All-test and person-only scores are almost identical.
- **Calculus is the weakest class.** Some sources box each tooth and others a whole region, which costs strict-IoU mAP more than it costs findings.
- **Do Qwen's labels help?** On v2 (2026-11-12), training without them lost 0.013 mAP50-95 on person-labelled test images (0.451 → 0.438), mostly on mouth ulcer and hypodontia. On v4 the difference is much smaller: 0.431 vs 0.428 on person test, 0.445 vs 0.429 on val. They still help slightly and do not hurt.

### 6.5 Weak spot: missing teeth outside Roboflow

The Roboflow hypodontia photos are rotated, saturated and high-contrast, and a missing tooth is a solid black hole. YOLO learned "black hole between teeth = hypodontia". In ordinary photos a gap shows pink gum or a soft shadow.

| Photos with a hypodontia box | Count | Before 7c (11-12 model) | After 7c (11-14 model) |
|---|---:|---:|---:|
| Roboflow test with hypodontia, unseen by either | 40 | 92.5% | 95.0% |
| Other conditions only (false alarms) | 338 | 0.6% | 1.5% |
| **Missing tooth, not Roboflow, never in YOLO's training** | **46** | **1 of 46** | **5 of 46** |

Before 7c, YOLO's training had 4,736 Roboflow hypodontia boxes and **zero** from dentalai, web or zenodo. The 89 gap-boxed photos moved it in the right direction but are few next to Roboflow's. **Next step:** repeat these photos about 5× in YOLO's train list and retrain.

---

## 7. Classifier vs YOLO: is stage 1 still needed?

*Added 2026-11-16.* YOLO can also answer "issue or not": any box means an issue. Scripts: `scripts/Daniel/clf_vs_yolo_score.py` and `clf_vs_yolo_analyse.py`.

**A fairness catch.** The cross-source sets are carved from the train split. The classifier never trains on them, but most of them are in YOLO's train list. So on cross-source photos the comparison favours YOLO.

### On the test set: YOLO slightly ahead, with different mistakes

| Rule | AUROC | Sensitivity | Specificity | Issue photos missed | Healthy flagged |
|---|---:|---:|---:|---:|---:|
| Classifier | 0.989 | 0.994 | 0.949 | 15 | 32 |
| YOLO | **0.998** | 0.989 | **0.977** | 25 | **14** |
| Issue if either fires | | **1.000** | 0.939 | **0** | 38 |

The error counts are not significantly different (McNemar p = 0.40), and the errors hardly overlap.

### On photos from other sources: the classifier generalises, YOLO does not

These 1,833 photos are all in YOLO's training set; the classifier has seen none of them.

| Issue photos from | Images | Classifier caught | YOLO caught |
|---|---:|---:|---:|
| zenodo | 542 | 0.982 | 0.983 |
| dentalai | 380 | 0.905 | 0.824 |
| kaggle_oral | 190 | **0.958** | **0.216** |
| web | 85 | 0.859 | 0.741 |
| **All** | **1,202** | **0.943** | **0.791** |

By condition, YOLO catches only **10%** of mouth ulcers and 26% of calculus outside Roboflow (classifier: 96% and 78%).

**Why.** YOLO's *ranking* is fine (AUROC 0.988 vs 0.983), but its *confidence* is not. On kaggle_oral its median top box is 0.15, below its 0.361 operating point; the classifier's median p(issue) is 0.99. YOLO learned its confidence on Roboflow photos. Lowering YOLO's confidence to about 0.07 matches the classifier's cross-source sensitivity but flags about 10% of healthy test photos, twice the classifier's rate.

### Decision

**Keep the classifier.**
1. It holds up on photos unlike Roboflow's, and app users' phone photos will not look like Roboflow's.
2. It learns from issue photos YOLO cannot use (missing teeth with no reliable box).
3. It is cheaper (about 7 ms vs 18 ms per image on GPU, including preprocessing).

**But it should not be the only gate.** Running both and flagging when *either* fires removes all 15 test misses and 31 of 69 cross-source misses, for about 1 point of specificity:

| Rule | Test sens. | Test spec. | Cross-source sens. | Cross-source spec. |
|---|---:|---:|---:|---:|
| Classifier only (current demo) | 0.994 | 0.949 | 0.943 | 0.962 |
| Either fires | **1.000** | 0.939 | **0.968** | 0.956 |

---

## 8. Demo app and SmileCheck SG prototype

### 8.1 Demo (`models/classifier/demo/Daniel/`)

A FastAPI single-page app (port 8712).

- **Reads everything from the checkpoints.** Backbone, preprocessing, temperature, threshold and the YOLO operating point come from the saved files. The "About this model" numbers are read from `test_metrics.json`, so they cannot go stale.
- **Mouth gate (`mouth_gate.py`).** The classifier only ever saw intraoral close-ups, so a selfie or a random picture would get a confident but meaningless score. The gate uses:
  - CLIP zero-shot to tell intraoral / face / other, and whether the mouth is open;
  - OpenCV YuNet to find the mouth corners in a face photo, level the lips and crop the mouth.

  Photos with no usable mouth are rejected with a reason (no mouth, mouth closed, too small). Sanity check: 595 of 600 test images passed as intraoral (the 5 rejected were heavily blurred augmentations and a brochure scan). On 25 web photos, every face with an open mouth was cropped correctly.
- **Detector (`detector.py`).** YOLO runs on flagged photos at full resolution. Boxes between 0.15 and the operating confidence are shown as "weaker candidates" behind a toggle.
- **A flagged photo is always outlined.** If YOLO finds nothing confident, its 3 strongest guesses are drawn dashed and labelled as hints. On the 2026-11-12 model this happened for 6 of 387 flagged issue photos but 17 of 27 flagged healthy ones. So "only guesses" usually means a classifier false alarm.
- **Privacy.** Images stay in memory during the request and are never written to disk.
- **Switches.** `DEMO_MOUTH_GATE=0` and `DEMO_DETECTOR=0` turn parts off; `DEMO_WEIGHTS` loads another checkpoint (e.g. the old 6-class one).

### 8.2 SmileCheck SG prototype (`prototype/Daniel/`)

A patient-facing page (port 8713) covering the brief's Detection, Understanding and Action gaps. `server.py` imports the demo's `app.py` rather than copying it, so both pages always run the same models and thresholds.

- **Photos and videos.** Every frame of a video goes through the mouth gate, the classifier and YOLO; progress streams to the page as NDJSON.
- **Combining results.**
  - A video is flagged when at least 10% of its frames (and at least 2) are flagged, so one blurry frame cannot flag a whole video.
  - The screening says *Needs a check* if any photo or video is flagged, because an issue may show from only one angle.
- **Understanding and action.** Each condition has a plain-language explanation, hand-drawn SVG illustrations, care products and a clinic map (Leaflet). The clinics and cost estimator are placeholders.
- **Classifier off switch** (`PROTO_CLASSIFIER=0` or `?classifier=off`). YOLO then screens every photo, which allows the comparison in section 7 to be seen in the app.

---

## 9. Engineering decisions

| Decision | Why |
|---|---|
| **Everything resumable.** Qwen answers are appended to JSONL as they arrive; trained runs are never retrained unless asked; YOLO resumes from `last.pt` | Long runs (Qwen labelling, a day of YOLO training) survive crashes, sleep and kernel restarts |
| **Thresholds applied when answers are read**, not when they are requested | Qwen's confidence or box rules can change without new requests |
| **Terminal runner** for the Qwen notebook (`run_qwen_pipeline.py`) | Long runs don't depend on VS Code keeping the kernel alive; Windows kept awake |
| **Images cached at ≤ 384 px** for the classifier; hard links instead of copies for YOLO folders | Fast loading; no extra disk space |
| **DataLoader worker processes** (Dataset classes in `classifier_data.py`) | Early runs were CPU-bound, with the GPU idle. On Windows, worker processes can only import Dataset classes from a `.py` file. A full 8-backbone sweep now takes about 50 minutes on an RTX 5070 Ti |
| **Restart kernels and close the demo before a long run** | One run hung for 48 minutes: Windows was out of committed memory (a 2-day-old kernel held 12.8 GB) and Python waited forever on a worker that never started |
| **Dataset zip with a fingerprint**, written to `.part` then renamed | Teammates can unzip and train; an interrupted zip is never left broken; unchanged data is not re-zipped |
| **Manual box viewer** (`view_yolo_boxes.py`) | Grid view with a review form to check Qwen's boxes by eye |

---

## 10. Limitations and responsible AI

- **Screening aid, not diagnosis.** "Normal" means none of the five screened conditions was seen. There is no gum disease class, although most Singapore adults have some gum disease. The app must say "none of the five screened conditions was detected" and always recommend check-ups.
- **The source confound remains.** Only data can remove it: healthy and diseased photos from the same cameras. Quote `xsrc_test` next to `test`.
- **Cross-source tests are narrow.** Outside Roboflow, issue photos are mostly caries. Ulcers, calculus and missing teeth on ordinary photos are barely tested (11–46 photos for missing teeth).
- **Label noise.** About 28% of YOLO's train images have Qwen labels (box IoU ≈ 0.19 against people's). The 7c gap boxes were checked only by a second Qwen request. Val/test condition labels are all human; most val/test healthy images were judged by Qwen.
- **No external validation.** Public datasets and web photos, not Singapore patients, and no demographic data (age, ethnicity, gum pigmentation). Real-world sensitivity, especially for early, subtle disease, is likely lower than measured.
- **Wording.** "Hypodontia" is congenital, but most photos show extracted or lost teeth, so the app should say "gap / missing tooth".
- **Silent misses.** A "normal" from stage 1 skips YOLO; this is why the threshold targets sensitivity and why the "either fires" rule is recommended.
- **Safety and privacy.** Emergency signs (swelling, bleeding, trauma) should be screened before the model and routed to 995 / A&E. Photos are health data under the PDPA: process in memory, keep nothing without consent, and never retrain on them without separate consent.

---

## 11. Lessons learned

1. **Check the data before the model.** Every large gain came from data: de-duplicating Kaggle (phase 1), the source confound (cross-source sets), and the label fixes (0.884 → 0.950 cross-source macro-F1). Eight backbones from 4 M to 28 M parameters landed within 0.01 of each other.
2. **A test set can be solved without the skill it is meant to measure.** The test set rewarded recognising Roboflow photos. A second test set that cannot be solved that way was needed.
3. **"Not labelled" is not "healthy".** Both label bugs (unannotated zenodo, unconfirmed web images) turned unchecked photos into healthy examples.
4. **Use an LLM labeller with guard rails.** Conservative prompts, evidence before the label, measured box accuracy, a second request to check its own boxes, and never in val/test.
5. **Pick the rule before seeing the results.** Fixed selection rules with tie margins avoided picking a large model for a difference inside the noise. They also explain why the choice changed when yolo11m's lead grew beyond the margin.
6. **Look at what the model gets wrong.** The missing-tooth problem was found by trying the demo on a real photo, and the label bugs by looking at the classifier's most confident errors.

---

## 12. Open items

| Item | Next step |
|---|---|
| Missing teeth outside Roboflow (both models) | Over-sample the non-Roboflow missing-tooth photos (classifier) and repeat the 89 gap-boxed photos ~5× in YOLO's train list |
| Flag a photo when either model fires | Run YOLO on every photo in the demo (+0.6 points test sensitivity, +2.5 cross-source, about −1 point specificity) |
| YOLO's low confidence on non-Roboflow photos | Over-sample non-Roboflow photos, or choose `conf` on a val set that includes them |
| yolo11m false alarms on kaggle Normal photos (7 of 59) | Look at the 7 photos: real false alarms or missed labels? |
| End-to-end score | Per-condition box scores for the full cascade |
| Report section 4 | Update to the 2026-11-16 yolo11m (this document already uses it) |
| Weights | `.pt` files are not in git; share them separately |

---

## 13. Where things are

| What | Path |
|---|---|
| Dataset zip / unzipped | `data/processed/qwen_labels/Daniel/oral_dataset.zip`, `data/raw/oral_dataset/oral_dataset/` |
| Per-image filter decisions | `data/processed/qwen_labels/Daniel/labels_filtered.csv` |
| Qwen answer caches | `data/interim/Daniel/qwen_pipeline/` |
| Notebooks | `models/classifier/notebooks/Daniel/` (Qwen pipeline, classifier), `models/yolo_caries/notebooks/Daniel/` (YOLO) |
| Scripts | `models/classifier/scripts/Daniel/` (scraper, Qwen runner, box viewer, classifier vs YOLO) |
| Classifier results | `models/classifier/results/Daniel/` (6-class at the top level; `binary_v1/`, `binary_v2/`, `binary/` = v3) |
| Classifier weights | `models/classifier/weights/Daniel/binary/best.pt` (v3); 6-class `best.pt` at the top level |
| YOLO results | `models/yolo_caries/results/Daniel/` (`comparison_val.csv`, `test_metrics.json`, `figures/`) |
| YOLO weights | `models/yolo_caries/weights/Daniel/best.pt` + `best.json` (yolo11m); `backup_20261112/` |
| Split manifests | `manifests/Daniel/` |
| Demo / prototype | `models/classifier/demo/Daniel/`, `prototype/Daniel/` |
| Detailed report | `docs/Daniel/report_classifier_yolo_qwen.md` |

**Figures for slides:** `eda_source_by_class.png` (the confound), `backbone_comparison.png`, `calibration_threshold.png`, `gradcam_cross-source_*.png` (classifier); `eda_classes_and_boxes.png`, `train_curves_and_val_per_class.png`, `test_image_level_tradeoff.png`, `test_normal_false_alarms.png` (YOLO).
