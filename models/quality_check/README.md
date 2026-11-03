# Quality check

Gate that rejects unusable photos (blurry, too dark, not intraoral) before they reach the classifier or detector.

- Training / thresholds: `notebooks/`
- Inference: `src/predict.py`
- Weights: `weights/` (gitignored)
