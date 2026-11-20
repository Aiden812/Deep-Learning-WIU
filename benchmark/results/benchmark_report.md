# External benchmark results (2026-11-19)

Photos no team model trained on (copies removed). Primary = **bench** part. 95% CIs in brackets (bootstrap over patients / photo clusters). Each model uses its **own** saved threshold.

| model | owner | what | photos predicted |
|---|---|---|---|
| aiden_gate_densenet | Aiden | DenseNet-121 gate | 5154 |
| aiden_gate_densenet_cal | Aiden | DenseNet-121 gate; threshold re-picked on MetaDent train part (0.503) | 5154 |
| aiden_gate_dinov2 | Aiden | DINOv2 ViT-B/14 gate | 5154 |
| aiden_gate_dinov2_cal | Aiden | DINOv2 ViT-B/14 gate; threshold re-picked on MetaDent train part (0.508) | 5154 |
| aiden_yolo11s | Aiden | YOLO11s on Daniel's v4 data | 5154 |
| aiden_yolo11s_cal | Aiden | YOLO11s on Daniel's v4 data; threshold re-picked on MetaDent train part (0.016) | 5154 |
| brandon_cls_v11 | Brandon | EfficientNet-B0 gate, v1.1 data | 5338 |
| brandon_cls_v11_cal | Brandon | EfficientNet-B0 gate, v1.1 data; threshold re-picked on MetaDent train part (0.314) | 5338 |
| brandon_cls_v12 | Brandon | EfficientNet-B0 gate, v1.2 data | 5338 |
| brandon_cls_v12_cal | Brandon | EfficientNet-B0 gate, v1.2 data; threshold re-picked on MetaDent train part (0.298) | 5338 |
| brandon_cls_v13 | Brandon | EfficientNet-B0 gate, v1.2 + external train parts, source-balanced | 5154 |
| brandon_yolo_v11 | Brandon | YOLO11n 640, v1.1 data | 5338 |
| brandon_yolo_v11_cal | Brandon | YOLO11n 640, v1.1 data; threshold re-picked on MetaDent train part (0.034) | 5338 |
| brandon_yolo_v12 | Brandon | YOLO11n 640, v1.2 data | 5338 |
| brandon_yolo_v12_cal | Brandon | YOLO11n 640, v1.2 data; threshold re-picked on MetaDent train part (0.112) | 5338 |
| brandon_yolo_v13 | Brandon | YOLO11n 960, v1.2 + external train parts, fine-tuned from v1.2 | 5154 |
| daniel_cls_6class | Daniel | EfficientNet-B0 6-class condition classifier | 5154 |
| daniel_cls_6class_cal | Daniel | EfficientNet-B0 6-class condition classifier; threshold re-picked on MetaDent train part (0.990) | 5154 |
| daniel_cls_binary | Daniel | binary gate | 5154 |
| daniel_cls_binary_cal | Daniel | binary gate; threshold re-picked on MetaDent train part (0.138) | 5154 |
| daniel_yolo | Daniel | YOLO12m 640 (no sidecar, so conf 0.25) | 5154 |
| daniel_yolo_cal | Daniel | YOLO12m 640 (no sidecar, so conf 0.25); threshold re-picked on MetaDent train part (0.045) | 5154 |
| junwei_yolo11n | Junwei | YOLO11n 640, 100 ep (conf from best.json sidecar) | 5154 |
| junwei_yolo11n_cal | Junwei | YOLO11n 640, 100 ep (conf from best.json sidecar); threshold re-picked on MetaDent train part (0.024) | 5154 |
| team_keras_6class | team | Keras EfficientNetB0 6-class baseline | 5338 |
| team_keras_6class_cal | team | Keras EfficientNetB0 6-class baseline; threshold re-picked on MetaDent train part (0.996) | 5338 |

Not scored yet (no weights on this laptop): aiden_yolo11n.


## 1. Problem vs healthy (MetaDent)

Problem = caries, plaque/calculus, discoloration, missing teeth or ulcer; healthy = no finding at all. Photos whose only findings are outside our 5 conditions (gums, wear, braces, fillings) are left out. **DS1 = hospital photos (fair: same camera for both groups); DS3 = web photos** (healthy ones are often stock smiles, so style can give the answer away).


### DS1: 94 problem / 20 healthy photos; size-only cheat tree balanced acc 0.48

| model | AUROC | balanced acc | sensitivity | specificity |
|---|---|---|---|---|
| aiden_gate_densenet | 0.586 (0.44–0.72) | 50% (37–63) | 46% (35–56) | 55% (31–76) |
| aiden_gate_densenet_cal | 0.586 (0.44–0.71) | 63% (56–71) | 32% (22–42) | 95% (83–100) |
| aiden_gate_dinov2 | 0.564 (0.44–0.67) | 64% (59–68) | 28% (20–36) | 100% (100–100) |
| aiden_gate_dinov2_cal | 0.564 (0.44–0.67) | 64% (60–68) | 28% (19–37) | 100% (100–100) |
| aiden_yolo11s | 0.878 (0.77–0.96) | 64% (60–69) | 29% (20–38) | 100% (100–100) |
| aiden_yolo11s_cal | 0.878 (0.76–0.96) | 80% (69–90) | 85% (77–92) | 75% (54–93) |
| brandon_cls_v11 | 0.581 (0.44–0.73) | 54% (42–65) | 43% (32–53) | 65% (43–87) |
| brandon_cls_v11_cal | 0.581 (0.42–0.72) | 55% (42–68) | 61% (50–71) | 50% (27–73) |
| brandon_cls_v12 | 0.538 (0.39–0.67) | 50% (37–61) | 35% (26–44) | 65% (41–86) |
| brandon_cls_v12_cal | 0.538 (0.40–0.68) | 55% (44–65) | 70% (61–80) | 40% (20–62) |
| brandon_cls_v13 | 0.882 (0.81–0.94) | 69% (64–74) | 37% (28–47) | 100% (100–100) |
| brandon_yolo_v11 | 0.824 (0.73–0.91) | 63% (59–68) | 27% (18–36) | 100% (100–100) |
| brandon_yolo_v11_cal | 0.824 (0.72–0.91) | 74% (63–84) | 78% (69–86) | 70% (50–90) |
| brandon_yolo_v12 | 0.833 (0.73–0.91) | 66% (59–72) | 37% (28–46) | 95% (82–100) |
| brandon_yolo_v12_cal | 0.833 (0.74–0.91) | 79% (71–86) | 63% (53–73) | 95% (83–100) |
| brandon_yolo_v13 | 0.937 (0.88–0.98) | 88% (82–94) | 82% (74–89) | 95% (84–100) |
| daniel_cls_6class | 0.678 (0.58–0.78) | 59% (47–71) | 73% (64–82) | 45% (21–69) |
| daniel_cls_6class_cal | 0.678 (0.56–0.78) | 68% (59–76) | 46% (36–56) | 90% (75–100) |
| daniel_cls_binary | 0.475 (0.33–0.62) | 54% (41–65) | 44% (34–54) | 65% (42–86) |
| daniel_cls_binary_cal | 0.475 (0.33–0.62) | 44% (35–54) | 68% (59–78) | 20% (5–38) |
| daniel_yolo | 0.849 (0.76–0.93) | 68% (63–73) | 36% (26–47) | 100% (100–100) |
| daniel_yolo_cal | 0.849 (0.75–0.92) | 80% (70–87) | 69% (59–78) | 90% (75–100) |
| junwei_yolo11n | 0.789 (0.68–0.89) | 57% (54–61) | 14% (8–21) | 100% (100–100) |
| junwei_yolo11n_cal | 0.789 (0.68–0.89) | 69% (59–80) | 63% (53–73) | 75% (54–94) |
| team_keras_6class | 0.713 (0.61–0.80) | 65% (54–76) | 65% (55–75) | 65% (44–85) |
| team_keras_6class_cal | 0.713 (0.61–0.81) | 53% (51–55) | 5% (1–10) | 100% (100–100) |

### DS3: 308 problem / 170 healthy photos; size-only cheat tree balanced acc 0.56

| model | AUROC | balanced acc | sensitivity | specificity |
|---|---|---|---|---|
| aiden_gate_densenet | 0.770 (0.73–0.81) | 69% (65–73) | 73% (68–78) | 65% (57–72) |
| aiden_gate_densenet_cal | 0.770 (0.73–0.81) | 70% (66–74) | 60% (55–65) | 81% (74–86) |
| aiden_gate_dinov2 | 0.827 (0.79–0.87) | 76% (73–80) | 75% (70–79) | 78% (72–84) |
| aiden_gate_dinov2_cal | 0.827 (0.79–0.86) | 76% (72–80) | 75% (70–80) | 78% (71–84) |
| aiden_yolo11s | 0.799 (0.75–0.84) | 61% (58–64) | 26% (22–31) | 96% (93–99) |
| aiden_yolo11s_cal | 0.799 (0.76–0.84) | 73% (69–78) | 77% (72–82) | 69% (63–76) |
| brandon_cls_v11 | 0.807 (0.76–0.85) | 73% (69–76) | 61% (56–67) | 84% (78–90) |
| brandon_cls_v11_cal | 0.807 (0.77–0.85) | 71% (67–75) | 69% (63–74) | 74% (67–80) |
| brandon_cls_v12 | 0.771 (0.73–0.81) | 70% (66–74) | 59% (54–65) | 81% (74–86) |
| brandon_cls_v12_cal | 0.771 (0.73–0.82) | 70% (66–75) | 75% (70–80) | 65% (58–73) |
| brandon_cls_v13 | 0.819 (0.78–0.85) | 72% (69–75) | 54% (49–60) | 90% (85–94) |
| brandon_yolo_v11 | 0.767 (0.72–0.81) | 61% (58–65) | 31% (25–35) | 92% (88–96) |
| brandon_yolo_v11_cal | 0.767 (0.72–0.81) | 72% (67–76) | 76% (71–80) | 68% (60–75) |
| brandon_yolo_v12 | 0.804 (0.76–0.84) | 69% (66–73) | 51% (46–57) | 88% (83–93) |
| brandon_yolo_v12_cal | 0.804 (0.76–0.85) | 73% (69–77) | 69% (64–74) | 76% (70–82) |
| brandon_yolo_v13 | 0.795 (0.75–0.84) | 74% (69–78) | 73% (68–77) | 75% (68–81) |
| daniel_cls_6class | 0.704 (0.66–0.75) | 58% (55–62) | 92% (89–95) | 24% (18–30) |
| daniel_cls_6class_cal | 0.704 (0.65–0.75) | 67% (62–72) | 69% (64–74) | 65% (57–71) |
| daniel_cls_binary | 0.747 (0.70–0.79) | 68% (64–72) | 57% (52–63) | 79% (73–86) |
| daniel_cls_binary_cal | 0.747 (0.70–0.79) | 71% (67–75) | 74% (69–79) | 68% (62–76) |
| daniel_yolo | 0.818 (0.78–0.86) | 64% (61–67) | 31% (26–37) | 96% (94–99) |
| daniel_yolo_cal | 0.818 (0.78–0.86) | 75% (71–78) | 64% (59–69) | 85% (80–90) |
| junwei_yolo11n | 0.787 (0.74–0.83) | 61% (58–63) | 25% (20–30) | 96% (93–99) |
| junwei_yolo11n_cal | 0.787 (0.75–0.83) | 71% (66–75) | 65% (60–70) | 76% (70–83) |
| team_keras_6class | 0.565 (0.51–0.62) | 52% (49–55) | 88% (85–92) | 15% (10–21) |
| team_keras_6class_cal | 0.565 (0.51–0.62) | 53% (50–56) | 17% (13–22) | 89% (84–93) |

## 2. Each condition, photo level

Models that name conditions: AUROC of that condition's score (photos with it vs photos without it). Gates (one yes/no score): share of photos with that condition they flag. MetaDent = DS1 + DS3 bench; AlphaDent caries = dentist-checked photos with / without caries (both from clinic patients).

| model | Caries (AlphaDent) | Caries | Calculus/plaque | Discoloration | Missing teeth |
|---|---|---|---|---|---|
| aiden_gate_densenet | flags 70% (65–76) | flags 69% (59–79) | flags 65% (56–73) | flags 65% (58–72) | flags 82% (73–89) |
| aiden_gate_densenet_cal | flags 56% (49–62) | flags 62% (51–73) | flags 51% (42–60) | flags 50% (43–57) | flags 70% (61–78) |
| aiden_gate_dinov2 | flags 59% (52–65) | flags 69% (59–78) | flags 59% (51–68) | flags 62% (55–69) | flags 83% (75–90) |
| aiden_gate_dinov2_cal | flags 59% (52–66) | flags 69% (59–78) | flags 59% (51–68) | flags 62% (56–69) | flags 83% (75–89) |
| aiden_yolo11s | AUROC 0.679 (0.60–0.76) | AUROC 0.813 (0.77–0.85) | AUROC 0.660 (0.61–0.71) | AUROC 0.708 (0.67–0.75) | AUROC 0.696 (0.64–0.75) |
| aiden_yolo11s_cal | AUROC 0.679 (0.60–0.76) | AUROC 0.813 (0.78–0.85) | AUROC 0.660 (0.61–0.71) | AUROC 0.708 (0.67–0.75) | AUROC 0.696 (0.64–0.75) |
| brandon_cls_v11 | flags 62% (55–68) | flags 60% (50–71) | flags 54% (46–62) | flags 60% (53–66) | flags 71% (62–80) |
| brandon_cls_v11_cal | flags 70% (64–76) | flags 69% (57–79) | flags 65% (57–74) | flags 67% (61–73) | flags 80% (71–87) |
| brandon_cls_v12 | flags 64% (58–71) | flags 69% (58–79) | flags 49% (40–58) | flags 49% (42–56) | flags 71% (62–80) |
| brandon_cls_v12_cal | flags 92% (87–95) | flags 91% (84–96) | flags 69% (61–77) | flags 70% (65–76) | flags 87% (79–93) |
| brandon_cls_v13 | flags 69% (63–74) | flags 57% (46–68) | flags 57% (48–65) | flags 49% (42–55) | flags 62% (51–71) |
| brandon_yolo_v11 | AUROC 0.680 (0.61–0.75) | AUROC 0.802 (0.76–0.85) | AUROC 0.681 (0.63–0.72) | AUROC 0.700 (0.66–0.74) | AUROC 0.564 (0.52–0.61) |
| brandon_yolo_v11_cal | AUROC 0.680 (0.61–0.75) | AUROC 0.802 (0.76–0.84) | AUROC 0.681 (0.63–0.73) | AUROC 0.700 (0.66–0.74) | AUROC 0.564 (0.52–0.61) |
| brandon_yolo_v12 | AUROC 0.677 (0.60–0.76) | AUROC 0.788 (0.75–0.83) | AUROC 0.679 (0.63–0.73) | AUROC 0.658 (0.62–0.70) | AUROC 0.673 (0.62–0.73) |
| brandon_yolo_v12_cal | AUROC 0.677 (0.60–0.75) | AUROC 0.788 (0.75–0.83) | AUROC 0.679 (0.63–0.73) | AUROC 0.658 (0.61–0.70) | AUROC 0.673 (0.62–0.73) |
| brandon_yolo_v13 | AUROC 0.792 (0.75–0.83) | AUROC 0.816 (0.76–0.87) | AUROC 0.692 (0.64–0.74) | AUROC 0.636 (0.59–0.68) | AUROC 0.689 (0.64–0.75) |
| daniel_cls_6class | AUROC 0.502 (0.44–0.57) | AUROC 0.619 (0.55–0.68) | AUROC 0.707 (0.66–0.75) | AUROC 0.461 (0.41–0.51) | AUROC 0.472 (0.41–0.53) |
| daniel_cls_6class_cal | AUROC 0.502 (0.44–0.57) | AUROC 0.619 (0.55–0.68) | AUROC 0.707 (0.66–0.75) | AUROC 0.461 (0.42–0.51) | AUROC 0.472 (0.41–0.54) |
| daniel_cls_binary | flags 70% (64–75) | flags 67% (57–77) | flags 49% (41–57) | flags 51% (45–57) | flags 71% (62–79) |
| daniel_cls_binary_cal | flags 86% (81–90) | flags 83% (74–91) | flags 69% (61–76) | flags 69% (63–75) | flags 89% (82–95) |
| daniel_yolo | AUROC 0.689 (0.61–0.77) | AUROC 0.813 (0.77–0.85) | AUROC 0.621 (0.58–0.67) | AUROC 0.712 (0.67–0.75) | AUROC 0.697 (0.65–0.75) |
| daniel_yolo_cal | AUROC 0.689 (0.62–0.76) | AUROC 0.813 (0.77–0.85) | AUROC 0.621 (0.58–0.67) | AUROC 0.712 (0.67–0.75) | AUROC 0.697 (0.64–0.75) |
| junwei_yolo11n | AUROC 0.680 (0.61–0.75) | AUROC 0.789 (0.75–0.83) | AUROC 0.638 (0.59–0.69) | AUROC 0.690 (0.65–0.73) | AUROC 0.692 (0.64–0.74) |
| junwei_yolo11n_cal | AUROC 0.680 (0.61–0.75) | AUROC 0.789 (0.75–0.83) | AUROC 0.638 (0.59–0.68) | AUROC 0.690 (0.65–0.73) | AUROC 0.692 (0.64–0.74) |
| team_keras_6class | AUROC 0.497 (0.43–0.57) | AUROC 0.491 (0.42–0.56) | AUROC 0.700 (0.65–0.74) | AUROC 0.496 (0.45–0.54) | AUROC 0.417 (0.36–0.47) |
| team_keras_6class_cal | AUROC 0.497 (0.42–0.58) | AUROC 0.491 (0.43–0.57) | AUROC 0.700 (0.65–0.74) | AUROC 0.496 (0.45–0.54) | AUROC 0.417 (0.36–0.47) |

Positives in the bench part: AlphaDent caries 369/518, caries 86/967, calculus/plaque 127/967, discoloration 210/967, missing teeth 99/967 photos.

## 3. Caries boxes (AlphaDent, dentist outlines)

Dentist polygons for caries classes 1-6 merged into one Caries box each; abrasion, fillings and crowns are not boxed. AlphaDent outlines **only the lesion**, while our training boxes (Roboflow style) often cover the whole tooth, so the strict score (IoU >= 0.5) punishes box size as well as missed caries. The loose score (IoU >= 0.1) asks only "did a caries box land on the lesion?". Recall / precision at the model's own confidence cut.

| model | part | photos | caries lesions | AP50 (strict) | AP10 (loose) | lesions found (loose) | boxes on a lesion (loose) |
|---|---|---|---|---|---|---|---|
| aiden_yolo11s | bench | 518 | 1527 | 0.024 (0.02–0.03) | 0.105 (0.08–0.14) | 7% (6–10) | 24% (17–33) |
| aiden_yolo11s | all | 1319 | 3576 | 0.023 (0.02–0.03) | 0.117 (0.10–0.14) | 7% (6–8) | 27% (22–33) |
| aiden_yolo11s_cal | bench | 518 | 1527 | 0.024 (0.02–0.03) | 0.105 (0.08–0.14) | 42% (37–48) | 11% (9–14) |
| aiden_yolo11s_cal | all | 1319 | 3576 | 0.023 (0.02–0.03) | 0.117 (0.10–0.14) | 43% (40–46) | 12% (11–14) |
| brandon_yolo_v11 | bench | 518 | 1527 | 0.024 (0.02–0.03) | 0.111 (0.09–0.15) | 5% (4–7) | 25% (17–35) |
| brandon_yolo_v11 | all | 1319 | 3576 | 0.024 (0.02–0.03) | 0.127 (0.11–0.15) | 5% (4–6) | 31% (25–37) |
| brandon_yolo_v11_cal | bench | 518 | 1527 | 0.024 (0.02–0.03) | 0.111 (0.09–0.14) | 32% (27–37) | 16% (13–19) |
| brandon_yolo_v11_cal | all | 1319 | 3576 | 0.024 (0.02–0.03) | 0.127 (0.11–0.15) | 32% (29–36) | 17% (15–19) |
| brandon_yolo_v12 | bench | 518 | 1527 | 0.024 (0.02–0.03) | 0.115 (0.09–0.15) | 7% (5–9) | 27% (18–38) |
| brandon_yolo_v12 | all | 1319 | 3576 | 0.025 (0.02–0.03) | 0.128 (0.11–0.16) | 8% (6–9) | 31% (24–38) |
| brandon_yolo_v12_cal | bench | 518 | 1527 | 0.024 (0.02–0.03) | 0.115 (0.09–0.16) | 18% (14–21) | 23% (17–30) |
| brandon_yolo_v12_cal | all | 1319 | 3576 | 0.025 (0.02–0.03) | 0.128 (0.10–0.16) | 18% (16–21) | 25% (20–30) |
| brandon_yolo_v13 | bench | 518 | 1527 | 0.310 (0.28–0.35) | 0.457 (0.41–0.50) | 72% (68–76) | 26% (23–29) |
| daniel_yolo | bench | 518 | 1527 | 0.023 (0.02–0.03) | 0.107 (0.08–0.14) | 10% (8–12) | 25% (19–32) |
| daniel_yolo | all | 1319 | 3576 | 0.020 (0.02–0.03) | 0.116 (0.10–0.14) | 11% (9–12) | 30% (25–35) |
| daniel_yolo_cal | bench | 518 | 1527 | 0.023 (0.02–0.03) | 0.107 (0.08–0.14) | 29% (24–33) | 16% (13–19) |
| daniel_yolo_cal | all | 1319 | 3576 | 0.020 (0.02–0.03) | 0.116 (0.10–0.14) | 29% (26–32) | 17% (14–19) |
| junwei_yolo11n | bench | 518 | 1527 | 0.025 (0.02–0.04) | 0.104 (0.08–0.14) | 4% (3–6) | 22% (14–32) |
| junwei_yolo11n | all | 1319 | 3576 | 0.027 (0.02–0.04) | 0.119 (0.10–0.15) | 4% (3–5) | 27% (21–35) |
| junwei_yolo11n_cal | bench | 518 | 1527 | 0.025 (0.02–0.04) | 0.104 (0.08–0.14) | 33% (28–39) | 15% (12–19) |
| junwei_yolo11n_cal | all | 1319 | 3576 | 0.027 (0.02–0.04) | 0.119 (0.10–0.14) | 33% (30–37) | 17% (14–19) |

## 4. Mouth ulcers

Autooral: ulcer close-ups with dentist masks (every photo has an ulcer). SMART-OM: normal tongue, cheek and lip photos (no ulcer), scored **separately** because the two sets look different. A gate counts as catching an ulcer if it flags the photo at all.

| model | ulcer photos caught (Autooral) | ulcer box AP50 | normal soft-tissue photos wrongly flagged (SMART-OM) |
|---|---|---|---|
| aiden_gate_densenet | 99% (97–100) (any-problem flag) | – | 54% (50–58) (any-problem flag) |
| aiden_gate_densenet_cal | 97% (94–100) (any-problem flag) | – | 38% (35–42) (any-problem flag) |
| aiden_gate_dinov2 | 95% (88–100) (any-problem flag) | – | 47% (43–51) (any-problem flag) |
| aiden_gate_dinov2_cal | 95% (88–100) (any-problem flag) | – | 48% (43–52) (any-problem flag) |
| aiden_yolo11s | 32% (23–48) | 0.241 (0.16–0.34) | 1% (0–1) |
| aiden_yolo11s_cal | 93% (87–100) | 0.241 (0.16–0.34) | 6% (5–8) |
| brandon_cls_v11 | 95% (90–98) (any-problem flag) | – | 45% (41–48) (any-problem flag) |
| brandon_cls_v11_cal | 99% (97–100) (any-problem flag) | – | 61% (57–64) (any-problem flag) |
| brandon_cls_v12 | 56% (46–67) (any-problem flag) | – | 35% (31–39) (any-problem flag) |
| brandon_cls_v12_cal | 79% (71–89) (any-problem flag) | – | 62% (58–65) (any-problem flag) |
| brandon_cls_v13 | 98% (96–100) (any-problem flag) | – | 0% (0–0) (any-problem flag) |
| brandon_yolo_v11 | 17% (10–30) | 0.159 (0.10–0.24) | 2% (1–2) |
| brandon_yolo_v11_cal | 85% (76–96) | 0.159 (0.11–0.24) | 17% (15–20) |
| brandon_yolo_v12 | 72% (60–87) | 0.120 (0.07–0.21) | 1% (0–2) |
| brandon_yolo_v12_cal | 82% (71–95) | 0.120 (0.07–0.21) | 4% (3–5) |
| brandon_yolo_v13 | 96% (87–100) | 0.672 (0.58–0.77) | 2% (1–3) |
| daniel_cls_6class | 91% (85–96) | – | 4% (3–5) |
| daniel_cls_6class_cal | 91% (85–96) | – | 4% (3–5) |
| daniel_cls_binary | 99% (98–100) (any-problem flag) | – | 50% (47–54) (any-problem flag) |
| daniel_cls_binary_cal | 100% (100–100) (any-problem flag) | – | 71% (68–74) (any-problem flag) |
| daniel_yolo | 43% (30–62) | 0.368 (0.27–0.50) | 1% (0–2) |
| daniel_yolo_cal | 82% (74–94) | 0.368 (0.26–0.51) | 6% (5–8) |
| junwei_yolo11n | 36% (25–57) | 0.109 (0.06–0.19) | 1% (0–2) |
| junwei_yolo11n_cal | 80% (67–92) | 0.109 (0.06–0.19) | 6% (4–7) |
| team_keras_6class | 87% (80–95) | – | 36% (32–40) |
| team_keras_6class_cal | 87% (80–95) | – | 36% (32–40) |

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
