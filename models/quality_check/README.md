# Quality check

Gate that rejects unusable photos before they reach the classifier or detector. No trained model: simple
image measurements with PIL + numpy, so it runs on the CPU in about 50 ms per photo.

- Code: `src/predict.py` (`check(PIL image)` or `predict(path)`)
- Used by: `app/backend/main.py`
  - `POST /api/quality`: one photo, called as soon as it is added, so the page can tick or cross each item
  - `POST /api/predict`: checks every photo, leaves out the ones that fail, classifies the rest

## What is checked

Each check gives **pass**, **warn** (still used, but the patient is told the result may be less reliable),
**fail** (left out) or **skip** (not judged, because another problem explains it).

| Check | Measured as | Fail | Warn |
|---|---|---|---|
| Big enough | short side of the original, px | < 100 | < 160 |
| Bright enough | 90th-percentile brightness (are the teeth lit? a dark background is fine) | < 60 | < 95 |
| Not washed out | mean brightness | > 225 | > 205 |
| No flash glare | share of pure-white pixels | never | > 15% |
| In focus | Laplacian variance, second-sharpest block of a 4×4 grid, on a 512 px copy | < 10 | < 35 |
| Clear detail | brightness standard deviation | < 14 | < 22 |
| Shows teeth and gums | share of gum / lip / tongue (pink-red) or tooth (pale) coloured pixels | < 15% | < 35% |

Only the cause is reported: a too-dark or washed-out photo skips focus, detail, colour and glare, and a dim
photo that looks colourless is blamed on the light.

## How the thresholds were set (2026-11-12)

120 photos (20 per class) from the classifier dataset, plus copies made darker, dimmer, washed out, blurred,
shrunk, greyscale and colour-swapped (red ↔ blue/green, as a stand-in for "not a mouth").

| Set | pass | warn | fail |
|---|---|---|---|
| Original photos | 95 | 20 | 5 (all under 100 px) |
| Very dark (×0.22) | 0 | 0 | 120 |
| Strong blur | 0 | 2 | 118 |
| Mild blur | 21 | 36 | 63 |
| Colour-swapped | 0 | 6 | 114 |

Most warnings on the originals are "small photo": the dataset is mostly web images around 285 px.

## Known limits

- "Shows teeth and gums" is a colour test. A close-up of skin or a pink object can pass it. A small trained
  "is this a mouth?" model would fix that.
- Thresholds were tuned on the classifier dataset and synthetic damage, not on real phone photos from users.
  Re-check them once there are real test photos.
