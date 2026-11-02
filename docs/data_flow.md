# Data flow

```
user photo
  → quality_check  ──(fail)──→ ask for retake
  → classifier  +  yolo_caries
  → shared/conditions.yaml (urgency, procedures)
  → cost/calculator.py (subsidised estimate)
  → genai (RAG explanation + agent: nearby CHAS clinics, products)
  → guardrails
  → app response
```

## Training data

```
data/raw/*  → scripts (convert, dedup)  → data/interim/  → data/processed/  (splits recorded in manifests/)
```
