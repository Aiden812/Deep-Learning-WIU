# External benchmark results (2026-11-17)

Photos no team model trained on (copies removed). Primary = **bench** part. 95% CIs in brackets (bootstrap over patients / photo clusters). Each model uses its **own** saved threshold.

| model | owner | what | photos predicted |
|---|---|---|---|
| brandon_cls_v11 | Brandon | EfficientNet-B0 gate, v1.1 data | 5338 |
| brandon_cls_v12 | Brandon | EfficientNet-B0 gate, v1.2 data | 5338 |
| brandon_yolo_v11 | Brandon | YOLO11n 640, v1.1 data | 5338 |
| brandon_yolo_v12 | Brandon | YOLO11n 640, v1.2 data | 5338 |
| team_keras_6class | team | Keras EfficientNetB0 6-class baseline | 5338 |

Not scored yet (no weights on this laptop): aiden_gate_dinov2, aiden_gate_densenet, aiden_yolo11n, aiden_yolo11s, daniel_cls_binary, daniel_cls_6class, daniel_yolo.


## 1. Problem vs healthy (MetaDent)

Problem = caries, plaque/calculus, discoloration, missing teeth or ulcer; healthy = no finding at all. Photos whose only findings are outside our 5 conditions (gums, wear, braces, fillings) are left out. **DS1 = hospital photos (fair: same camera for both groups); DS3 = web photos** (healthy ones are often stock smiles, so style can give the answer away).


### DS1: 94 problem / 20 healthy photos; size-only cheat tree balanced acc 0.48

| model | AUROC | balanced acc | sensitivity | specificity |
|---|---|---|---|---|
| brandon_cls_v11 | 0.581 (0.43–0.73) | 54% (41–65) | 43% (33–53) | 65% (42–84) |
| brandon_cls_v12 | 0.538 (0.39–0.67) | 50% (37–62) | 35% (26–45) | 65% (40–85) |
| brandon_yolo_v11 | 0.824 (0.71–0.91) | 63% (59–68) | 27% (18–37) | 100% (100–100) |
| brandon_yolo_v12 | 0.833 (0.74–0.92) | 66% (59–73) | 37% (27–47) | 95% (84–100) |
| team_keras_6class | 0.713 (0.60–0.81) | 65% (53–75) | 65% (55–74) | 65% (44–85) |

### DS3: 308 problem / 170 healthy photos; size-only cheat tree balanced acc 0.56

| model | AUROC | balanced acc | sensitivity | specificity |
|---|---|---|---|---|
| brandon_cls_v11 | 0.807 (0.77–0.84) | 73% (68–76) | 61% (56–67) | 84% (78–89) |
| brandon_cls_v12 | 0.771 (0.73–0.81) | 70% (66–74) | 59% (54–65) | 81% (74–86) |
| brandon_yolo_v11 | 0.767 (0.72–0.81) | 61% (58–65) | 31% (26–36) | 92% (88–96) |
| brandon_yolo_v12 | 0.804 (0.76–0.84) | 69% (66–73) | 51% (45–56) | 88% (82–93) |
| team_keras_6class | 0.565 (0.51–0.62) | 52% (49–55) | 88% (85–92) | 15% (10–21) |

## 2. Each condition, photo level

Models that name conditions: AUROC of that condition's score (photos with it vs photos without it). Gates (one yes/no score): share of photos with that condition they flag. MetaDent = DS1 + DS3 bench; AlphaDent caries = dentist-checked photos with / without caries (both from clinic patients).

| model | Caries (AlphaDent) | Caries | Calculus/plaque | Discoloration | Missing teeth |
|---|---|---|---|---|---|
| brandon_cls_v11 | flags 62% (55–67) | flags 60% (50–71) | flags 54% (45–63) | flags 60% (53–66) | flags 71% (61–80) |
| brandon_cls_v12 | flags 64% (59–70) | flags 69% (59–78) | flags 49% (41–58) | flags 49% (42–55) | flags 71% (62–80) |
| brandon_yolo_v11 | AUROC 0.680 (0.60–0.75) | AUROC 0.802 (0.76–0.85) | AUROC 0.681 (0.63–0.73) | AUROC 0.700 (0.66–0.74) | AUROC 0.564 (0.52–0.61) |
| brandon_yolo_v12 | AUROC 0.677 (0.59–0.75) | AUROC 0.788 (0.74–0.83) | AUROC 0.679 (0.63–0.73) | AUROC 0.658 (0.62–0.70) | AUROC 0.673 (0.62–0.73) |
| team_keras_6class | AUROC 0.497 (0.43–0.57) | AUROC 0.491 (0.42–0.56) | AUROC 0.700 (0.66–0.74) | AUROC 0.496 (0.45–0.54) | AUROC 0.417 (0.37–0.47) |

Positives in the bench part: AlphaDent caries 369/518, caries 86/967, calculus/plaque 127/967, discoloration 210/967, missing teeth 99/967 photos.

## 3. Caries boxes (AlphaDent, dentist outlines)

Dentist polygons for caries classes 1-6 merged into one Caries box each; abrasion, fillings and crowns are not boxed. AlphaDent outlines **only the lesion**, while our training boxes (Roboflow style) often cover the whole tooth, so the strict score (IoU >= 0.5) punishes box size as well as missed caries. The loose score (IoU >= 0.1) asks only "did a caries box land on the lesion?". Recall / precision at the model's own confidence cut.

| model | part | photos | caries lesions | AP50 (strict) | AP10 (loose) | lesions found (loose) | boxes on a lesion (loose) |
|---|---|---|---|---|---|---|---|
| brandon_yolo_v11 | bench | 518 | 1527 | 0.024 (0.02–0.03) | 0.111 (0.08–0.15) | 5% (4–7) | 25% (18–35) |
| brandon_yolo_v11 | all | 1319 | 3576 | 0.024 (0.02–0.03) | 0.127 (0.11–0.15) | 5% (4–7) | 31% (25–37) |
| brandon_yolo_v12 | bench | 518 | 1527 | 0.024 (0.02–0.04) | 0.115 (0.09–0.16) | 7% (5–9) | 27% (18–38) |
| brandon_yolo_v12 | all | 1319 | 3576 | 0.025 (0.02–0.03) | 0.128 (0.11–0.16) | 8% (6–9) | 31% (24–38) |

## 4. Mouth ulcers

Autooral: ulcer close-ups with dentist masks (every photo has an ulcer). SMART-OM: normal tongue, cheek and lip photos (no ulcer), scored **separately** because the two sets look different. A gate counts as catching an ulcer if it flags the photo at all.

| model | ulcer photos caught (Autooral) | ulcer box AP50 | normal soft-tissue photos wrongly flagged (SMART-OM) |
|---|---|---|---|
| brandon_cls_v11 | 95% (90–98) (any-problem flag) | – | 45% (41–48) (any-problem flag) |
| brandon_cls_v12 | 56% (46–67) (any-problem flag) | – | 35% (32–39) (any-problem flag) |
| brandon_yolo_v11 | 17% (10–29) | 0.159 (0.11–0.24) | 2% (1–2) |
| brandon_yolo_v12 | 72% (60–87) | 0.120 (0.07–0.21) | 1% (0–2) |
| team_keras_6class | 87% (80–95) | – | 36% (32–40) |

158 Autooral photos (43 patients), 788 SMART-OM photos (153 patients).

## 5. Speed and size (CPU, one photo at a time)

| model | median ms/photo | 90th pct ms | weights MB |
|---|---|---|---|
| brandon_cls_v11 | 65 | 83 | 16.3 |
| brandon_cls_v12 | 60 | 81 | 16.3 |
| brandon_yolo_v11 | 120 | 142 | 5.5 |
| brandon_yolo_v12 | 124 | 208 | 5.5 |
| team_keras_6class | 249 | 277 | 29.0 |

## Data used

| dataset | bench | excluded | train |
|---|---|---|---|
| alphadent | 518 | 1 | 801 |
| autooral | 158 | 0 | 262 |
| metadent | 967 | 764 | 857 |
| smartom | 788 | 1 | 803 |

Excluded: MetaDent DS2 = the Mendeley set our Kaggle training data came from: 481; face portrait, not a mouth photo: 128; not a real photo of a person: 79; copy of training photo: 78.
