# PolyMarket Predictor

Offline machine learning pipeline for collecting Polymarket snapshots, preparing reusable datasets, training category-aware models, and backtesting predictions.

## What it does

- Downloads raw market snapshots from the public Gamma API.
- Prepares a reusable tabular dataset from those snapshots.
- Builds temporal features from earlier snapshots of the same market when available.
- Builds one model per category, plus a global fallback model.
- Blends category models with the global model instead of hard switching.
- Adds richer market, event, and text-summary features on top of the raw price state.
- Calibrates predicted probabilities on held-out validation data before saving the model bundle.
- Evaluates models with a chronological train/validation split.
- Runs a simple backtest on validation data using configurable edge thresholds.
- Predicts `YES` probabilities for open markets and compares them to the current market price.

## Project layout

- `polymarket_predictor/api/client.py`: Public API clients for Gamma and CLOB.
- `polymarket_predictor/datasets/data.py`: Raw snapshot fetch and dataset preparation.
- `polymarket_predictor/datasets/features.py`: Feature extraction and category normalization.
- `polymarket_predictor/datasets/backfill.py`: Historical market and price-series backfill helpers.
- `polymarket_predictor/ml/model.py`: Lightweight numpy model implementations.
- `polymarket_predictor/ml/models.py`: Pluggable model adapters and calibration.
- `polymarket_predictor/ml/metrics.py`: Validation and backtest metrics.
- `polymarket_predictor/ml/pipeline.py`: Training and prediction orchestration.
- `polymarket_predictor/cli.py`: Command-line interface.
- `polymarket_predictor/ui/gui.py`: Desktop GUI for fetch/prepare/train/predict and model comparisons.
- `polymarket_predictor/ui/gui_utils.py`: Comparison helpers used by the GUI.
- `polymarket_predictor/review/snapshotting.py`: Saved prediction snapshot and review workflow.
- `polymarket_predictor/datasets/taxonomy.json`: Editable category mapping and aliases.

## Install

```bash
pip install -r requirements.txt
```

## Launch GUI

```bash
python -m launch_gui
```

The GUI lets you:

- control fetch, prepare, train, and predict parameters without typing commands
- train multiple model families in one run
- compare saved training metrics side by side
- compare predictions from multiple artifact directories on the same live markets

## Fetch Raw Snapshots

```bash
python -m polymarket_predictor.cli fetch --output artifacts/raw_closed.jsonl --closed --max-pages 20 --page-size 200
python -m polymarket_predictor.cli fetch --output artifacts/raw_open.jsonl --max-pages 5 --page-size 200 --order volume24hr
python -m polymarket_predictor.cli fetch --output artifacts/raw_open.jsonl --max-pages 5 --page-size 200 --order volume24hr --append
```

Use `--append` on recurring fetches so the same file accumulates market history over time.

## Prepare Dataset

```bash
python -m polymarket_predictor.cli prepare --input artifacts/raw_closed.jsonl --output artifacts/prepared_training_data.csv
```

## Backfill Historical Data

```bash
python -m polymarket_predictor.cli backfill-markets --output artifacts/raw_closed_history.jsonl --max-pages 50 --page-size 200 --append
python -m polymarket_predictor.cli backfill-prices --market-snapshots artifacts/raw_closed_history.jsonl --output artifacts/price_history.jsonl --interval 1d --fidelity 60 --max-markets 500
python -m polymarket_predictor.cli build-horizon-dataset --price-history artifacts/price_history.jsonl --output artifacts/horizon_training_data.csv --horizon-hours 24 168 720
```

This backfill flow creates synthetic training rows representing what the market looked like 24 hours, 7 days, or 30 days before resolution.

If a large `backfill-markets` run times out, use the chunked six-month script instead:

```bash
python scripts/backfill_last_6_months.py --artifact-dir artifacts/six_month_backfill
```

That script:

- walks closed markets page by page instead of doing one huge pull
- stops automatically once it reaches markets older than roughly six months
- retries timed-out page and price-history requests with backoff
- resumes safely from existing JSONL outputs
- builds the horizon dataset and trains the selected model families

## Train

```bash
python -m polymarket_predictor.cli train --artifact-dir artifacts --dataset artifacts/prepared_training_data.csv --validation-fraction 0.2 --model-type logistic
python -m polymarket_predictor.cli train --artifact-dir artifacts --dataset artifacts/horizon_training_data.csv --validation-fraction 0.2 --model-type boosted_trees
```

This stores:

- `artifacts/model_bundle.json`
- `artifacts/training_metrics.json`

## Predict

```bash
python -m polymarket_predictor.cli predict --artifact-dir artifacts --limit 50 --category politics
python -m polymarket_predictor.cli predict --artifact-dir artifacts --history-input artifacts/raw_open.jsonl --limit 50
```

Example CSV export:

```bash
python -m polymarket_predictor.cli predict --artifact-dir artifacts --output predictions.csv
```

## Notes

- The code targets the public Gamma API documented at [Polymarket docs](https://docs.polymarket.com/api-reference/introduction) and [List markets](https://docs.polymarket.com/api-reference/markets/list-markets).
- Historical price backfill uses the CLOB `prices-history` endpoint with CLOB token IDs. The official timeseries docs note that `interval` should not be combined with `startTs`/`endTs`, so the backfill flow uses bounded requests without `interval` and only falls back to interval-only requests when needed.
- Training uses only resolved binary markets where the outcome can be inferred from final `outcomePrices`.
- Validation is chronological rather than random, so the reported metrics are out-of-sample on newer markets.
- If a category lacks enough training examples, prediction falls back to the global model.
