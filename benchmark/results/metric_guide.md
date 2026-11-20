# How to read the benchmark report

## Which file does what

| step | file | what it does in plain words |
|---|---|---|
| 1 | build_manifest.py | Makes the exam paper: lists every usable photo, its dentist answer, and whether it is bench (exam) or train |
| 2 | run.py | Gives every photo to each model and saves the model's raw answers to predictions/<model>.csv |
| 2a | adapters/ (called by run.py) | Translators: each one opens a different model type and makes it answer in the same format |
| 3 | score.py | Marks the answers against the dentist labels and writes benchmark_report.md + results.json |

## What each number means

| result | what it judges against (one sentence) |
|---|---|
| photos predicted | How many benchmark photos the model gave an answer for (all of them = it ran fully). |
| (_cal) model name | Same model, but its yes/no cut-off was re-picked on the training half of MetaDent instead of using its own saved one. |
| numbers in brackets | The range the score would likely fall in if we had drawn a different set of patients (95% confidence). |
| size-only cheat tree | A dumb model that only sees photo width and height; if it scores well, photo shape alone gives the answer away (0.5 = no cheat possible). |
| AUROC | If you show the model one problem photo and one healthy photo, how often it gives the problem photo the higher score (0.5 = coin flip, 1.0 = perfect). |
| balanced acc | The average of sensitivity and specificity, so a model can't look good just by always saying "problem". |
| sensitivity | Of the photos a dentist says have a problem, the share the model flagged. |
| specificity | Of the photos a dentist says are healthy, the share the model correctly left alone. |
| Condition: AUROC | For one condition (e.g. caries), how well that condition's score ranks photos with it above photos without it. |
| Condition: flags X% | For yes/no models with no per-condition score: of photos that have this condition, the share it flagged as "problem" at all. |
| part: bench / all | bench = the frozen exam photos only (main score); all = bench plus the training half (extra photos, only fair for models not trained on them). |
| caries lesions | How many dentist-outlined caries spots are in those photos. |
| AP50 (strict) | How well the model's boxes match the dentist's outlines when a box must overlap the lesion by at least half (punishes wrong box size). |
| AP10 (loose) | Same as AP50 but a box only needs to touch the lesion a little, so it asks "did it point at the right spot?". |
| lesions found (loose) | Of all dentist-marked caries spots, the share that got a model box on them. |
| boxes on a lesion (loose) | Of all boxes the model drew, the share that actually landed on a real caries spot (the rest are false alarms). |
| ulcer photos caught | Of ulcer photos (all have an ulcer), the share the model flagged. |
| ulcer box AP50 | How well the model's ulcer boxes match the dentist's ulcer outlines. |
| normal soft-tissue wrongly flagged | Of normal tongue/cheek/lip photos with no ulcer, the share the model wrongly flagged (lower is better). |
| median ms/photo | Typical time to check one photo on a normal CPU. |
| 90th pct ms | The slow-case time: 9 in 10 photos finish faster than this. |
| weights MB | File size of the model, which matters for putting it in an app. |
| bench / train / excluded | How many photos went to the exam, to the may-train-on pile, or were thrown out (copies of our data, faces, non-photos). |
