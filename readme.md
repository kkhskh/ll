# Trexquant - Earnings Return Prediction

## Files
- `baseline_experiments.py` — local diagnostic script with walk-forward CV
- `advanced_experiments.py` — cross-sectional normalization + LightGBM experiments  
- `colab_train.py` — **GPU training script for Google Colab (H100)**
- `demo-submission-notebook.ipynb` — Kaggle submission notebook

## Quickstart (Colab)

See the Colab section below.

## Local run

```bash
python baseline_experiments.py --train-path train.csv --test-path test.csv
```

## Colab (H100) run

See setup commands below.
