# Oral screening pipeline: data, classifier and detector (Daniel)

**IT2226 AI Innovator and Deep Learning Project.** Status as of 2026-11-14.

This report collects what was built, measured and found for the two-stage oral screening pipeline: the dataset built with Qwen, the stage-1 classifier ("is there an issue?") and the stage-2 YOLO detector ("where, and what?"). The notebooks hold the full detail; this document is the summary with the numbers in one place.

| Part | Notebook | Main outputs |
|---|---|---|
| Data cleaning and labelling with Qwen | `models/classifier/notebooks/Daniel/qwen_clean_and_label.ipynb` | `data/processed/qwen_labels/Daniel/oral_dataset.zip` (unzipped to `data/raw/oral_dataset/oral_dataset/`) |
| Stage 1: binary classifier | `models/classifier/notebooks/Daniel/oral_condition_classifier.ipynb` | `models/classifier/weights/Daniel/binary/best.pt`, `models/classifier/results/Daniel/binary/` |
| Stage 2: YOLO detector | `models/yolo_caries/notebooks/Daniel/yolo_oral_detector.ipynb` | `models/yolo_caries/weights/Daniel/best.pt` + `best.json`, `models/yolo_caries/results/Daniel/` |
| Demo app | `models/classifier/demo/Daniel/app.py` | loads both checkpoints above |

---

## 1. Key findings

1. **The data has a source confound.** In val/test, every image with a condition is a 640×640 `roboflow_oral` export and every healthy image comes from another source. A model can score well by recognising Roboflow photos. Every result below is therefore reported twice: on the normal test set (an upper bound) and on a **cross-source** set (`xsrc_test`) of non-Roboflow photos that contain both labels.
2. **Stage 1 (classifier)** is MobileNetV3-Large, picked by a rule fixed before training. Test macro-F1 is 0.975. On other-source photos (`xsrc_test`) it is **0.944**, AUROC 0.978. At the operating threshold (95% sensitivity on val and `xsrc_val`), sensitivity is 0.928 and specificity 0.963 on `xsrc_test`. Eight backbones from 4 M to 28 M parameters land within about 0.01 of each other, so the smallest and fastest one was chosen (8 ms per image on CPU).
3. **Stage 2 (YOLO)** is YOLO11-small, retrained on 2026-11-14 on the current build. Test mAP50 is 0.814 and mAP50-95 0.430. At image level it boxes 99.0% of issue photos, with a 2.3% false-alarm rate on healthy photos. The scores on person-labelled test images are the same (0.431), so the Qwen-judged Normal images are not distorting the score. On 708 test photos that neither it nor the previous model trained on, the two models tie (mAP50-95 0.401 vs 0.397). In the 2026-11-12 ablation, training without the Qwen-labelled images was worse (mAP50-95 0.438 vs 0.451), so they help despite their noise.
4. **Hypodontia (missing teeth) is the weak spot of both stages outside Roboflow.** Almost every missing-tooth example came from `roboflow_oral`, whose photos are heavily processed: a gap there is a solid black hole. On ordinary photos with a missing tooth:
   - the classifier caught 4–5 of 13 held-out photos;
   - before 7c, YOLO boxed the gap on **3 of 109** (2.8%), against 94% of Roboflow test photos;
   - after retraining with the 7c gap boxes, YOLO boxes **5 of the 46** such photos it never trained on (11%; the previous model 1 of 46).

   Two Qwen passes (7b and 7c, section 2.4) added 143 confirmed missing-tooth photos and boxes for 89 of them. Both stages have been retrained on them, each with only a small gain (classifier 2 → 3 of 11 held-out photos, YOLO 1 → 5 of 46). The new photos are few next to the Roboflow ones; over-sampling them is the next thing to try.
5. **Most of the gains came from cleaning labels.** Two rules made images "Normal" without anyone having checked them (section 2.3). Fixing them raised the classifier's `xsrc_test` macro-F1 from 0.884 (ResNet-50 selected then, `xsrc_test` of 1,594 images) to 0.950 (MobileNetV3, 1,166 images). The test sets also changed with the fix, so read this as a large improvement rather than an exact number.

---

## 2. Data: building the dataset with Qwen

### 2.1 Sources

| Source | What it is | Labels it brings | Route in the notebook |
|---|---|---|---|
| `roboflow_oral` | Roboflow oral-diseases export, 640×640, many augmented copies | Human boxes for all five conditions (gingivitis dropped) | `gt`: no Qwen request |
| `dentalai`, `zenodo` | Clinical photos (zenodo up to ~8,700 px) | Human **caries** boxes only | `gt_check`: Qwen does a light check for the other four conditions |
| `kaggle_oral` | Folders per condition | The folder names the condition | `known`: Qwen must box the named condition |
| `mendeley_noncaries` | Caries-free clinical close-ups | "No caries" | `label`: Qwen labels and boxes, caries not offered |
| Web (`data/raw/external`) | 21,000 scraped images | The search term | `web`: Qwen screens, labels and boxes |

The model was `qwen3.8-flash-next-radiance` (earlier answers from `qwen3.8-flash-next-mxfp4` were reused), served by llama-swap. Thinking was always off. Every answer is cached as JSONL per step and model in `data/interim/Daniel/qwen_pipeline/`, so a run can be interrupted and resumed, and changing a threshold needs no new requests.

**Web images.** De-duplication by perceptual hash took 21,000 images down to 12,968 unique ones: 3,843 photos had been scraped under more than one class. The screen accepted only **665 (5%)**: real teeth, unobstructed, a close-up, a photograph and not a collage.

### 2.2 How Qwen was kept honest

- **Conservative prompts.** "A false alarm is worse than a miss." Each finding gives its evidence first, then the label, then a confidence. Only `high` findings are kept.
- **Look-alike guide.** The prompt explains how to tell calculus (material added), caries (material lost, a hole) and discoloration (colour only, smooth surface) apart. It also lists things that are not conditions (shadows, fillings, slightly yellow teeth).
- **Measured effect of the label prompt** (130 kaggle / mendeley images):

  | Measure | Old prompt | New prompt |
  |---|---|---|
  | kaggle `Normal` labelled Normal | 12% | 70% |
  | Caries on kaggle `Normal` | 48% | 3% |
  | Caries on caries-free mendeley | 37% | 8% |
  | Caries on kaggle discoloration | 49% | 10% |

- **Box coordinates.** These were measured against human boxes (32 images). Qwen's boxes are on a 0–1000 scale. With the right reading, mean IoU is only 0.19 and precision 0.31. The misplacement comes from Qwen's localisation, not a coordinate bug. **Qwen's boxes are rough**, and val/test never use them (section 2.5).

### 2.3 Filtering (section 7 of the notebook)

Boxes were checked by eye, and the main problems became rules:

| Dropped | Why | Count (current build) |
|---|---|---|
| Qwen `Hypodontia` boxes (except kaggle's Hypodontia folder) | Mostly false: on side views Qwen marked hidden or out-of-frame teeth as missing (39% of upper-left/right views vs 4% of lower-front) | 258 boxes |
| Every mendeley image Qwen flagged with a condition | Its calculus and hypodontia findings were too unreliable | 4,515 images |
| Qwen boxes covering > 60% of the image | Told a condition must be boxed, Qwen sometimes boxed the whole photo | 54 boxes |
| Light-check boxes wrapping a human caries box | Qwen re-boxed the marked caries as discoloration | 83 boxes |
| Human boxes thinner than 0.5% of the image | ~3 px at 640, too small to learn from | 588 boxes |
| kaggle copies of roboflow photos | Same photo; keep the copy with human boxes | 2,773 images |

**Label fixes (7a, 2026-11-11).** An image may only be "Normal" if someone looked and found nothing. The classifier's most confident "errors" on Normal images showed two rules that broke this:

| Problem | Fix |
|---|---|
| zenodo's annotations cover only about a third of its photos, and "no annotation file" had been read as "no caries". About 3,400 never-checked photos became Normal, and many show clear decay | Only zenodo photos with an annotation file count (4,039 dropped) |
| A web image found by a condition search became Normal whenever Qwen found nothing at high confidence | It stays Normal only if every search that found it was a Normal search (75 dropped) |

**Background cap.** Healthy images are capped at 10% of YOLO's train set (a detector that sees too many empty images learns to stay quiet). The rest are not lost: the classifier gets all of them.

### 2.4 Hypodontia re-check (7b) and gap boxes (7c), 2026-11-13

**Why.** The demo called a photo with an obvious missing front tooth "normal". The cause was in the data: nearly all hypodontia examples were Roboflow photos. Qwen's other hypodontia findings had been dropped together with their unreliable boxes, so clinical close-ups like mendeley's were all healthy. The classifier had learned that a clinical-looking photo means healthy. It flagged **0.1%** of mendeley photos where Qwen had seen a gap, against 97% of Roboflow's.

**7b: one stricter question per image.** "Is a tooth missing?" A tooth counts as missing only if there is a gap inside the row, wide enough for a tooth, with nothing else explaining it: a hidden or overlapping tooth, camera angle, edge of the photo, a narrow space, a child's lost baby tooth, or a false tooth. Only the 2,013 images whose hypodontia finding came from Qwen were asked (no full relabel). The answer took about 0.8 s per image, with no failures.

| Dataset | Asked | Missing tooth confirmed |
|---|---:|---:|
| mendeley_noncaries | 1,776 | 62 |
| dentalai | 66 | 47 |
| external_clean (web) | 107 | 26 |
| zenodo | 49 | 8 |
| kaggle_oral | 15 | 0 |
| **total** | **2,013** | **143** |

109 of the 143 had no other boxes. They became **classifier-only** issue images (`has_issue = 1`, `in_yolo = 0`, no label file).

**7c: boxes for YOLO, checked by a second request instead of a person.** Qwen's original boxes on these photos were looked at by eye: about 6 in 10 sat on the gap, and the rest were on a tooth, a crack, a narrow space or the wrong spot. 7c asks again:
1. **Box request.** The prompt passes Qwen's own 7b note on where the gap is and lists what is not a gap. Boxes must also pass shape rules: at most 15% of the image, not a thin sliver, and at most 6 per photo.
2. **Check request.** The photo is sent again with the box drawn in green, and Qwen is asked what is inside it: a gap, a tooth, a broken tooth, a narrow space, and so on.

| Result | Count |
|---|---:|
| Boxes drawn | 144 |
| Pass the shape rules | 132 |
| Rejected by the check (16 on a tooth, 9 on a broken tooth) | 25 |
| **Kept** | **107, on 106 photos** |

A look at the kept boxes: about **8 in 10 sit on the gap** (up from about 6 in 10). The doubtful ones are mostly small boxes on narrow spaces between teeth. No person reviewed them; that was a deliberate time trade-off.

Effect on the dataset:
- 63 classifier-only photos joined YOLO's train set with gap boxes.
- 26 photos already in YOLO got their unboxed gap labelled.
- **17 classifier-only photos are held out of YOLO** (seeded 20%) to test YOLO on missing teeth outside Roboflow.
- 29 photos have no reliable box and stay classifier-only.

### 2.5 Split

- **Copies stay together.** Copies of one photo (Roboflow augmentations, perceptual-hash near-duplicates, kaggle re-uploads of Roboflow photos) are grouped and always share a split.
- **val/test hold only images whose condition labels were drawn by people.** Issue images are Roboflow groups, stratified by their rarest class; healthy images are a share of the Normal groups of every source. So the scores measure accuracy against people, not agreement with Qwen. `label_by` records who decided each label. `data_person.yaml` gives the same exam restricted to fully person-labelled images.
- **Stable split (2026-11-13).** Val/test used to be drawn by shuffling photo-group numbers, and those numbers shift whenever an image is added. Adding the 109 hypodontia images would have reshuffled about 60% of the healthy val/test images. Groups are now ordered by a seeded hash of their smallest MD5, so adding images no longer moves the others (checked: 0 val and 2 test images moved for the 109, none for the 7c boxes). Switching to this ordering redrew val/test once: about 2,300 of the 2,936 new test images had been in the old train split. **A model trained before 2026-11-13 must not be scored on the current val/test.**

### 2.6 Current dataset (built 2026-11-13, with 7c)

| | train | val | test |
|---|---:|---:|---:|
| Calculus (images) | 3,661 | 369 | 401 |
| Caries | 10,738 | 830 | 837 |
| Hypodontia | 2,628 | 311 | 335 |
| Mouth_Ulcer | 4,699 | 574 | 543 |
| Tooth_Discoloration | 7,832 | 937 | 898 |
| Normal (no boxes, YOLO) | 2,603 | 609 | 621 |
| **YOLO images** | **26,032** | **2,941** | **2,936** |
| Classifier: issue | 23,475 | 2,332 | 2,315 |
| … of which classifier-only missing teeth | 46 | 0 | 0 |
| Classifier: healthy | 4,600 | 609 | 621 |
| Labelled entirely by people (YOLO) | 18,664 | 2,392 | 2,374 |

33,952 images in total, in a 3.64 GB zip. Backups of earlier builds are in `data/backups/` (`oral_dataset_v2_20261111`, `oral_dataset_v3_20261113`).

| Build | Date | What changed | Trained on it |
|---|---|---|---|
| v2 | 2026-11-11 | 7a label fixes | classifier v2, YOLO 2026-11-12 (backed up) |
| v3 | 2026-11-13 | + 7b classifier-only missing teeth, stable split (val/test redrawn once) | classifier v3 (current) |
| v4 (current) | 2026-11-13 | + 7c gap boxes (YOLO only; classifier data and split identical to v3) | YOLO 2026-11-14 (current) |

---

## 3. Stage 1: binary classifier ("issue or not")

### 3.1 Setup

- **Task.** Does the photo show any of the five screened conditions? Flagged photos go to YOLO; a "normal" result skips it.
- **Cross-source check sets.** `xsrc_val` and `xsrc_test` (about 1,180 images each) are carved by photo group out of the non-Roboflow training sources, which contain both labels. They are stratified by source × label. `xsrc_val` counts in model selection; `xsrc_test` is used once at the end.
- **Recipe B** (all backbones):
  - balanced sampler: healthy and issue drawn equally often; within issue, Roboflow and other sources get half each; every photo group is equally likely;
  - strong augmentation, including blur and JPEG to mimic phone photos, with hue jitter kept tiny because colour is diagnostic;
  - label smoothing 0.1; 224×224 squash (no centre crop, so lip ulcers are not cut off); at most 15 epochs of 12,000 samples.
- **Baseline A.** ConvNeXt-Tiny with a uniform sampler and flips only.
- **Selection rule, fixed before training.** Validation score = mean macro-F1 of clean val, a phone-quality copy of val, and `xsrc_val`. Runs within 0.01 of the best are tied; the fastest on CPU wins.
- **Operating point.** One temperature is fitted on val + `xsrc_val`. The threshold is the highest that still gives **95% sensitivity on both** val and `xsrc_val`, because a missed condition is silent in the app.

### 3.2 Results (v3, trained on the current split)

| Run | Params (M) | CPU ms | val score | test macro-F1 | `xsrc_test` macro-F1 [95% CI] | `xsrc_test` AUROC |
|---|---:|---:|---:|---:|---|---:|
| B EfficientNet-V2-S | 20.2 | 25.9 | 0.968 | 0.976 | 0.947 [0.934, 0.961] | 0.977 |
| B ResNet-50 | 23.5 | 21.6 | 0.966 | 0.975 | 0.946 [0.932, 0.959] | 0.976 |
| B EfficientNet-B0 | 4.0 | 10.9 | 0.965 | 0.976 | 0.945 [0.930, 0.959] | 0.975 |
| **B MobileNetV3-Large (selected)** | **4.2** | **8.3** | **0.965** | **0.975** | **0.944 [0.929, 0.957]** | **0.978** |
| B ConvNeXt-Tiny | 27.8 | 30.4 | 0.964 | 0.969 | 0.948 [0.935, 0.960] | 0.976 |
| B Swin-T | 27.5 | 31.1 | 0.962 | 0.974 | 0.947 [0.932, 0.960] | 0.980 |
| B DenseNet-121 | 7.0 | 23.5 | 0.961 | 0.971 | 0.947 [0.934, 0.960] | 0.968 |
| A ConvNeXt-Tiny (baseline) | 27.8 | 28.4 | 0.961 | 0.974 | 0.939 [0.925, 0.952] | 0.986 |

Confidence intervals come from 1,000 bootstrap resamples of photo groups. All B runs overlap, so **architecture and size make no measurable difference here**, and the 4 M-parameter MobileNet was chosen on speed. Recipe B over baseline A on ConvNeXt-Tiny gives +0.009 `xsrc_test` macro-F1, inside the CI. B's clearer advantage is robustness:

| Degradation (`xsrc_test`, macro-F1) | A ConvNeXt-Tiny | B ConvNeXt-Tiny | B MobileNetV3 |
|---|---:|---:|---:|
| original | 0.939 | 0.948 | 0.944 |
| low-res 160 px + JPEG 60 | 0.897 | 0.933 | 0.928 |
| blur (r = 2) | **0.854** | 0.943 | 0.943 |
| JPEG q = 20 | 0.897 | 0.944 | 0.947 |
| bright ×1.6 | 0.911 | 0.924 | 0.929 |

Under blur, the baseline's sensitivity on `xsrc_test` drops to 0.751; B's stays at 0.92–0.93.

**At the operating threshold** (temperature 0.706, threshold 0.729, set by `xsrc_val`; val alone would have given 0.974):

| Set | Sensitivity | Specificity | Macro-F1 | ECE (raw → calibrated) |
|---|---:|---:|---:|---|
| test | 0.992 | 0.947 | 0.973 | 0.043 → 0.010 |
| test, person-labelled | 0.992 | 0.881 | 0.897 | 0.050 → 0.018 |
| `xsrc_test` | 0.928 | 0.963 | 0.945 | 0.036 → 0.028 |

**Per source (`xsrc_test`, threshold 0.729):**

| Source | Images (issue / healthy) | Sensitivity | Specificity | AUROC |
|---|---|---:|---:|---:|
| dentalai | 190 / 92 | 0.895 | 0.848 | 0.914 |
| external_clean | 43 / 6 | 0.837 | 1.000 | 0.961 |
| kaggle_oral | 95 / 15 | 0.937 | 0.933 | 0.936 |
| mendeley_noncaries | 8 / 455 | **0.125** | 0.996 | 0.929 |
| zenodo | 271 / 6 | 0.985 | 0.333 | 0.814 |

The mendeley row is the hypodontia problem (section 3.4): its 8 issue images are mostly missing teeth.

### 3.3 v2 → v3: a fair comparison

The split was redrawn between v2 and v3, so the numbers in the two `experiments.csv` files are not comparable. Both models were scored on the **1,031 test + `xsrc_test` images that neither trained on**, each at its own threshold:

| Set | Images | v2 sens / spec | v3 sens / spec | AUROC v2 → v3 |
|---|---:|---|---|---|
| All held-out | 1,031 | 0.971 / 0.936 | 0.975 / 0.947 | 0.976 → 0.979 |
| Roboflow test | 614 | 0.997 / 0.919 | 0.997 / 0.936 | 0.970 → 0.980 |
| Other sources | 417 | 0.918 / 0.953 | 0.929 / 0.957 | 0.971 → 0.974 |

So v3 is slightly better overall, mostly on specificity.

### 3.4 Hypodontia in the classifier

Issue photos caught on the same held-out set, per condition:

| Condition | Source | Images | v2 | v3 |
|---|---|---:|---:|---:|
| Caries | other sources | 152 | 0.967 | 0.980 |
| Tooth discoloration | other sources | 21 | 0.905 | 0.905 |
| Mouth ulcer | other sources | 9 | 1.000 | 1.000 |
| Hypodontia | Roboflow | 40 | 0.975 | 1.000 |
| **Hypodontia** | **other sources** | **13** | **0.308** | **0.385** |
| … of which the 7b photos | other sources | 11 | 0.182 | 0.273 |

- **Caught:** smiles and implant photos.
- **Still missed:** close-up clinical photos with a clear gap, mostly scored at P(issue) ≈ 0.02 against the 0.729 threshold.
- **Some progress:** v3 ranks mendeley's missing-tooth photos above its healthy ones far more often (AUROC on mendeley 0.69 → 0.94). Only one clears the threshold, but two more moved up (to 0.65 and 0.18).

**Why the gain is small.** Only about 87 of the new photos are in training, among 23,000 issue images. The sampler balances healthy against issue and Roboflow against the rest, not conditions. So each epoch draws about 20 of them, against thousands of near-identical healthy close-ups. Suggested next step (not done): over-sample non-Roboflow issue photos, or condition-balance within the issue class, and retrain MobileNet only (about 6 minutes).

### 3.5 Training speed note

Early runs were CPU-bound: the GPU sat idle while the main process decoded JPEGs. Image loading now runs in DataLoader worker processes (8 train, 6 eval, persistent across epochs). On Windows the Dataset classes must live in a `.py` module (`classifier_data.py`) so spawned workers can import them. A full 8-backbone sweep takes about 50 minutes on an RTX 5070 Ti.

One v3 run hung for 48 minutes without output. A stack dump showed the kernel blocked sending data to the 6th validation worker, which never started. Windows was nearly out of committed memory: a 2-day-old kernel held 12.8 GB, the demo app 4.5 GB, and each worker about 1.5 GB. On Windows, Python then waits forever with no error. **Restart the kernel, and close the demo and other kernels, before a long training run.** Lowering `num_workers` / `eval_workers` also helps.

---

## 4. Stage 2: YOLO detector ("what and where")

### 4.1 Setup

- **Models.** COCO-pretrained YOLO11 nano / small / medium, plus an ablation `yolo11s_person` that trains only on images labelled entirely by people (18,598 of 25,850).
- **Recipe.** 640 px, batch 32 (16 for medium), up to 80 epochs, early stopping with patience 15.
- **Augmentation.** Hue jitter almost off and saturation/brightness jitter reduced, because colour carries discoloration and calculus. No vertical flips or rotation.
- **Training copy.** Very large clinical photos are resized once to at most 1,024 px, so mosaic augmentation doesn't leave the GPU waiting on decoding.
- **Selection.** Highest val mAP50-95; runs within 0.005 are tied, and the fewest GFLOPs wins.

### 4.2 Results (current model: trained 2026-11-14 on build v4)

Only yolo11s was retrained on v4. The size comparison is from the 2026-11-12 sweep on v2; the val sets differ, so compare rows within a build only.

| Run | Build | Params (M) | GFLOPs | Epochs (best) | val mAP50 | val mAP50-95 |
|---|---|---:|---:|---|---:|---:|
| yolo11n | v2 | 2.6 | 6.6 | 80 (76) | 0.812 | 0.435 |
| yolo11s | v2 | 9.4 | 21.7 | 57 (42) | 0.821 | 0.444 |
| yolo11m | v2 | 20.1 | 68.3 | 62 (50) | 0.825 | 0.446 |
| yolo11s_person | v2 | 9.4 | 21.7 | 49 (35) | 0.797 | 0.426 |
| **yolo11s (current)** | **v4** | **9.4** | **21.7** | **78 (62)** | **0.825** | **0.445** |

On v2, yolo11m was within the 0.005 tie margin of yolo11s at 3× the compute, so yolo11s was chosen and is the only size retrained. Training took about 2.2 min per epoch on an RTX 5070 Ti; inference is about 2 ms per image on GPU.

**Test (yolo11s, operating confidence 0.317 = max mean F1 on val):**

| | All test images (2,936) | Person-labelled (2,374) |
|---|---:|---:|
| Precision / recall | 0.790 / 0.777 | 0.790 / 0.778 |
| mAP50 | 0.814 | 0.815 |
| mAP50-95 | 0.430 | 0.431 |

| Condition | AP50 | AP50-95 | Image recall | Image precision |
|---|---:|---:|---:|---:|
| Calculus | 0.644 | 0.299 | 0.835 | 0.852 |
| Caries | 0.812 | 0.437 | 0.897 | 0.938 |
| Hypodontia | 0.801 | 0.347 | 0.940 | 0.926 |
| Mouth_Ulcer | 0.922 | 0.488 | 0.976 | 0.987 |
| Tooth_Discoloration | 0.892 | 0.579 | 0.922 | 0.876 |

**Image level:**
- 99.0% of the 2,315 issue photos get at least one box.
- 2.3% of the 621 healthy photos get a false alarm (1.7% of the 59 person-labelled healthy photos).

**Findings:**
- **These numbers are lower than the previous model's (mAP50-95 0.450), but the test sets differ.** The previous model was scored on v2's test split; since the split was redrawn, about 2,200 of the current test photos were in its training set. Section 4.3 compares the two fairly.
- **Qwen's labels help** (2026-11-12 ablation on v2). Training without the Qwen-labelled images lost 0.013 mAP50-95 on person-labelled test images (0.451 → 0.438), most on mouth ulcer (−0.019) and hypodontia (−0.019). Not repeated on v4.
- **Qwen's Normal labels are fine for scoring.** All-test and person-only scores are almost identical.
- **Calculus is the weakest class** (AP50 0.64, image recall 0.84). Some sources box each tooth and others a whole region, which costs mAP50-95 more than it costs findings.

### 4.3 Previous vs current model, on photos neither trained on

708 photos of the current test split were not in the previous model's training set (378 with conditions, 330 healthy). Both models were scored on them, each at its own operating confidence (previous 0.357, current 0.317).

| | Previous (2026-11-12) | Current (2026-11-14) |
|---|---:|---:|
| Precision / recall | 0.713 / 0.717 | 0.721 / 0.717 |
| mAP50 | 0.742 | 0.734 |
| mAP50-95 | 0.397 | **0.401** |
| Hypodontia AP50 / AP50-95 | 0.539 / 0.204 | **0.565 / 0.234** |
| Caries AP50 | 0.829 | 0.798 |
| Calculus AP50 | 0.502 | 0.463 |
| Mouth ulcer / discoloration AP50 | 0.938 / 0.901 | 0.938 / 0.907 |
| Healthy photos with any box (330) | 1.5% | 2.4% |

The two models tie overall. Hypodontia is a little better and caries and calculus a little worse, but with 40 hypodontia photos and a few hundred of the rest, none of these differences is clearly beyond noise. The 708 photos are a harder mix than the full test set (mAP50-95 0.40 against 0.43).

### 4.4 Hypodontia: the Roboflow look

The 2026-11-12 model, before 7c, was run on hypodontia photos it never trained on:

| Photos | Count | Photos with a hypodontia box (conf ≥ 0.357) |
|---|---:|---:|
| Roboflow test (v2 split), with hypodontia | 397 | **94.0%** (box recall 0.79 at IoU 0.5) |
| Roboflow test, other conditions only | 2,050 | 1.3% (false alarms) |
| Healthy test photos | 599 | 0.2% |
| **7b missing-tooth photos, not Roboflow** | **109** | **2.8%** (5.5% even at conf 0.15) |

Training hypodontia boxes by source (before 7c): Roboflow 4,736, kaggle 48, dentalai / external / zenodo **0**. Compare caries: about 12,500 non-Roboflow boxes.

Looking at the photos explains it. The Roboflow hypodontia photos are rotated with black borders, strongly saturated and high-contrast, and the missing teeth appear as **solid black holes**. YOLO learned "black hole between teeth = hypodontia". In ordinary photos a gap shows pink gum or a soft shadow, and YOLO sees nothing, even on obvious smiles with a front tooth gone. Retraining on the same data cannot fix this; it needs ordinary photos with gap boxes, which is what 7c adds (89 photos with boxes, 17 held out for testing).

**After 7c (current model).** Photos with a hypodontia box, each model at its own confidence:

| Photos | Count | Previous | Current |
|---|---:|---:|---:|
| Roboflow test with hypodontia, not in either model's training | 40 | 92.5% | 95.0% |
| Test photos with other conditions only (false alarms) | 338 | 0.6% | 1.5% |
| **Missing tooth, not Roboflow: held out of YOLO in 7c** | **17** | **0%** | **11.8%** (2) |
| **Missing tooth, not Roboflow: no reliable gap box in 7c** | **29** | **3.4%** (1) | **10.3%** (3) |

The 46 non-Roboflow photos are mendeley 29, dentalai 9, external 7 and zenodo 1. The gap boxes moved YOLO in the right direction (1 → 5 of 46), but most missing teeth in ordinary photos are still missed: 89 boxed photos are few next to about 4,700 Roboflow hypodontia boxes. In the pipeline, these photos depend on the classifier to flag them, and YOLO will usually not mark the gap. Next step: repeat the 89 photos about 5× in YOLO's train list and retrain yolo11s, to see whether the limit is the amount of data or the photos themselves.

---

## 5. Limitations and responsible AI

- **Screening aid, not diagnosis.** "Normal" means none of the five screened conditions was seen. There is no gum disease class, although most Singapore adults have some gum disease. The app must say "none of the five screened conditions was detected" and always recommend check-ups.
- **The source confound remains.** Only data can remove it: healthy and diseased photos taken with the same cameras. `test` numbers are an upper bound; quote `xsrc_test` next to them.
- **`xsrc_test` is narrow.** Outside Roboflow, issue images are mostly caries (plus zenodo discoloration). Ulcers, calculus and hypodontia on non-Roboflow photos are barely tested; for hypodontia, 11–17 photos only show the direction.
- **YOLO's test set is Roboflow-only for conditions.** YOLO is not measured on clinical or phone photos with conditions, apart from the 46 missing-tooth photos in section 4.4.
- **Label noise.** About 28% of YOLO train images have Qwen labels. Qwen's boxes have IoU ≈ 0.19 against people's. The 7c gap boxes were checked only by a second Qwen request (about 8 in 10 good on inspection). Val/test condition labels are all human; most val/test healthy images were judged by Qwen.
- **No external validation.** Images are public datasets and web photos, not Singapore patients, and there is no demographic metadata (age, ethnicity, gum pigmentation). Real-world sensitivity, especially for early, subtle disease, is likely lower than measured.
- **Hypodontia wording.** "Hypodontia" is congenital, but most photos show extracted or lost teeth. The app should say "gap / missing tooth".
- **Silent misses.** A "normal" from stage 1 skips YOLO, which is why the threshold targets sensitivity.
- **App safety.** Emergency signs (swelling, bleeding, trauma) should be screened before the model. Photos are health data: process them in memory and don't keep them without consent.

---

## 6. Open items

| Item | Status | Next step |
|---|---|---|
| YOLO retrain on build v4 (7c gap boxes, current split) | **Done** (2026-11-14) | Test mAP50-95 0.430; ties the previous model on photos neither trained on (section 4.3). Previous model backed up in `models/yolo_caries/weights/Daniel/backup_20261112/`. |
| YOLO: missing teeth outside Roboflow | Small gain only | 1 → 5 of 46 photos (section 4.4). Repeat the 89 gap-boxed photos about 5× in the YOLO train list and retrain yolo11s only. |
| Classifier: missing teeth in clinical close-ups | Small gain only | Over-sample non-Roboflow issue photos / condition-balance the issue class; retrain MobileNet only |
| End-to-end pipeline score | To do | Both stages are now trained on the same split; score classifier → YOLO on the shared test set |
| Commit | Done | Merged with Aiden's `dev` and pushed (`484ec6c`). Weights (`.pt`) are not in git and have to be shared separately. |
| YOLO notebook saved outputs | Check | The per-source false-alarm table saved in `yolo_oral_detector.ipynb` still shows the 2026-11-12 run (599 healthy test photos); re-run those cells to refresh it. |

---

## 7. Where things are

| What | Path |
|---|---|
| Dataset zip / unzipped | `data/processed/qwen_labels/Daniel/oral_dataset.zip`, `data/raw/oral_dataset/oral_dataset/` (README inside) |
| Per-image filter decisions | `data/processed/qwen_labels/Daniel/labels_filtered.csv` (`drop_reason`, `label_by`, `split`) |
| Qwen answer caches | `data/interim/Daniel/qwen_pipeline/<step>__<model>.jsonl` |
| Classifier split manifest | `manifests/Daniel/binary_classifier_split.csv` |
| Classifier results / figures | `models/classifier/results/Daniel/binary/` (`experiments.csv`, `test_metrics.json`, `figures/`); v2 in `binary_v2/` |
| Classifier weights | `models/classifier/weights/Daniel/binary/best.pt` (v3); v2 in `binary_v2/` |
| YOLO results / figures | `models/yolo_caries/results/Daniel/` (`comparison_val.csv`, `test_metrics.json`, `figures/`); team log `models/yolo_caries/results/experiments.csv` |
| YOLO weights | `models/yolo_caries/weights/Daniel/best.pt` + `best.json` |
| Backups | `data/backups/oral_dataset_v2_20261111/`, `data/backups/oral_dataset_v3_20261113/`, `models/yolo_caries/weights/Daniel/backup_20261112/` |

Useful figures for slides:
- Classifier: `eda_source_by_class.png` (the confound), `backbone_comparison.png`, `calibration_threshold.png`, `gradcam_cross-source_*.png`.
- YOLO: `eda_classes_and_boxes.png`, `train_curves_and_val_per_class.png`, `test_image_level_tradeoff.png`, `test_normal_false_alarms.png`.
