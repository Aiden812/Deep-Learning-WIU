# Decisions log

One entry per decision

## YYYY-MM-DD — Title
- **Decision:**
- **Alternatives considered:**
- **Why:**
- **Owner:**

## 2026-11-05 — Condition classifier (Aiden): DenseNet-121 backbone
- **Decision:** Aiden's classifier uses DenseNet-121 (ImageNet-pretrained, two-stage fine-tune).
- **Alternatives considered:** EfficientNet-B0 (an earlier run of Aiden's on the Kaggle dataset, dropped), ResNet50 / ConvNeXt-Tiny / EfficientNet-B0 (Daniel's comparison).
- **Why:** Daniel already compares those three on the same task; a fourth architecture adds a comparison instead of duplicating work.
- **Owner:** Aiden

## 2026-11-05 — Roboflow oral-diseases split leaks copies of training photos
- **Decision:** Do not use Roboflow's train/valid/test split. Re-split by original photo.
- **Evidence:** Comparing every val/test image with all training images (DenseNet, then DINOv2 features), pairs that
  looked identical were cropped / rotated / recoloured copies of one photo, down to low similarity; only 10 of 1,831 val/test
  images had no close training copy. SIFT keypoint matching on a sample: 92-98% of Roboflow "photos" have a copy filed as a
  different photo. The source dataset already contained augmented copies under new names before Roboflow added 3 per training photo.
  The Kaggle oral dataset (used by that earlier EfficientNet run) is almost entirely contained in the Roboflow set (90-100% of its disease photos match),
  so it cannot serve as an external test set, and that run's test score (macro-F1 0.964) was most likely inflated by the same copies.
- **Alternatives considered:** keep Roboflow's split (scores would measure memory, not recognition); group with global
  embeddings only (DenseNet and DINOv2 similarities of copies and of different photos overlap, and grouping chains
  into one cluster holding 73-98% of images).
- **Why:** a test set must contain photos the model has never seen in any form.
- **Result** (`prepare_roboflow_multilabel.py`): copies grouped by DINOv2 candidates (20 per image) confirmed by SIFT +
  RANSAC. >= 40 matching keypoints = same original photo (checked by eye; at 20, groups chain different mouths:
  `models/classifier/results/Aiden/figures/copy_groups_*.png`). >= 10 keypoints links photos into split groups, which are
  split 70/15/15 (at <= 8 they chain into one group of 1,278+ photos). 9,155 Roboflow photos = 2,164 original photos.
  Val/test keep one image per split group, so they hold no copies of each other either: val 156 / test 119 images,
  24-79 positives per condition (60/20/20 was tried, but Gingivitis and Hypodontia gain nothing and training loses 14% of
  its photos, so 70/15/15 was kept). No train/val-test candidate pair has >= 10 matching keypoints. Most Gingivitis and
  Hypodontia photos are copies inside one 470-photo split group, which must stay in train. 424 pairs with 5-9 remain:
  random different photos reach up to ~20, so some are copies that slipped through
  (`cross_split_pairs_below_threshold.png`). Residual leakage is small but not zero.
  Run 1 kept one image per original photo instead (val/test 324/324, with weaker copies inside test): test macro AP 0.829.
  Run 2, on the stricter set: 0.859. Within-test copies were not inflating scores, so the stricter set is kept.
- **Owner:** Aiden

## 2026-11-05 — Condition classifier (Aiden): multi-label instead of single-label
- **Decision:** One sigmoid output per condition (BCE loss with pos_weight, per-condition thresholds tuned on validation).
  A photo's labels = classes of its Roboflow boxes + its folder class, unioned over all copies of the photo.
- **Alternatives considered:** (1) single-label on the re-split Roboflow data, dropping photos filed under two classes;
  (2) back to the Kaggle oral dataset with a stronger duplicate check (keeps a Normal class, but same data as Daniel, no gingivitis).
- **Why:** a third of photos have boxes for several conditions, and copies of one photo are filed under different classes
  (30% of Calculus photos also appear as Gingivitis): one mouth often shows several conditions, so single-label targets contradict each other.
- **Consequences:** no Normal class (the app can only say "no condition above threshold"); unboxed conditions count as absent,
  so some negatives are really unlabelled positives; the app's single-class `predict.py` contract needs a multi-label version.
- **Owner:** Aiden
