# Jun Wei: YOLO11n results

Two runs of YOLO11n with the same settings; only the image size changes (640 vs 960 px), 100 epochs each.
Code: `notebooks/JunWei/02_training_script.ipynb`. Scores: `results/experiments.csv` (rows `Junwei_yolo11n_*_100ep`).

| Folder | What it holds |
|---|---|
| `yolo11n_640_100ep/`, `yolo11n_960_100ep/` | Training: settings (`args.yaml`) and one row per epoch (`results.csv`) |
| `..._test/` | Final score on the held-out test set (PR curve, confusion matrix, one example) |
| `..._test_person/` | Same test set, only images labelled by people |
| `compare_640_vs_960.png` | Validation mAP50 and mAP50-95 of both runs per epoch, and the gap between them |

Final model (640 px run): `models/yolo_caries/weights/Junwei/best.pt`, with `best.json` (classes, image size, confidence cut-off 0.221 picked on val). Weights inside the run folders are gitignored.
