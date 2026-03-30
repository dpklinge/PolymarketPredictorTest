# PolyMarket Predictor

Offline machine learning pipeline for training category-aware Polymarket models from public APIs and generating predictions for live markets.

## What it does

- Downloads historical binary markets from the public Gamma API.
- Builds one logistic model per category, plus a global fallback model.
- Learns from resolved markets by using the final winning outcome as the training label.
- Predicts calibrated `YES` probabilities for open markets and compares them to the current market price.

## Project layout

- `polymarket_predictor/client.py`: Public API client for Polymarket Gamma.
- `polymarket_predictor/features.py`: Feature extraction and category normalization.
- `polymarket_predictor/model.py`: Lightweight numpy logistic regression.
- `polymarket_predictor/pipeline.py`: Training and prediction orchestration.
- `polymarket_predictor/cli.py`: Command-line interface.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m polymarket_predictor.cli train --artifact-dir artifacts --max-pages 20 --page-size 200
```

This stores:

- `artifacts/model_bundle.json`
- `artifacts/training_metrics.json`

## Predict

```bash
python -m polymarket_predictor.cli predict --artifact-dir artifacts --limit 50 --category politics
```

Example CSV export:

```bash
python -m polymarket_predictor.cli predict --artifact-dir artifacts --output predictions.csv
```

## Notes

- The code targets the public Gamma API documented at [Polymarket docs](https://docs.polymarket.com/api-reference/introduction) and [List markets](https://docs.polymarket.com/api-reference/markets/list-markets).
- Training uses only resolved binary markets where the outcome can be inferred from final `outcomePrices`.
- If a category lacks enough training examples, prediction falls back to the global model.
