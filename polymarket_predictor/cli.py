from __future__ import annotations

import argparse
import json

from .datasets.backfill import backfill_closed_markets, backfill_price_history, build_horizon_dataset
from .datasets.data import fetch_snapshots, prepare_dataset
from .ml.pipeline import predict_open_markets, train_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run category-aware Polymarket predictors.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch raw market snapshots from the API.")
    fetch_parser.add_argument("--output", required=True)
    fetch_parser.add_argument("--closed", action="store_true")
    fetch_parser.add_argument("--max-pages", type=int, default=25)
    fetch_parser.add_argument("--page-size", type=int, default=200)
    fetch_parser.add_argument("--order", default="updatedAt")
    fetch_parser.add_argument("--ascending", action="store_true")
    fetch_parser.add_argument("--append", action="store_true")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare a flat training dataset from raw snapshots.")
    prepare_parser.add_argument("--input", nargs="+", required=True)
    prepare_parser.add_argument("--output", required=True)

    backfill_markets_parser = subparsers.add_parser("backfill-markets", help="Backfill many closed markets into a JSONL file.")
    backfill_markets_parser.add_argument("--output", required=True)
    backfill_markets_parser.add_argument("--max-pages", type=int, default=50)
    backfill_markets_parser.add_argument("--page-size", type=int, default=200)
    backfill_markets_parser.add_argument("--order", default="updatedAt")
    backfill_markets_parser.add_argument("--ascending", action="store_true")
    backfill_markets_parser.add_argument("--append", action="store_true")

    backfill_prices_parser = subparsers.add_parser("backfill-prices", help="Backfill historical price series for closed markets.")
    backfill_prices_parser.add_argument("--market-snapshots", required=True)
    backfill_prices_parser.add_argument("--output", required=True)
    backfill_prices_parser.add_argument("--interval", default="1d")
    backfill_prices_parser.add_argument("--fidelity", type=int, default=60)
    backfill_prices_parser.add_argument("--max-markets", type=int, default=None)
    backfill_prices_parser.add_argument("--append", action="store_true")

    horizon_parser = subparsers.add_parser("build-horizon-dataset", help="Build horizon-based training rows from price history.")
    horizon_parser.add_argument("--price-history", required=True)
    horizon_parser.add_argument("--output", required=True)
    horizon_parser.add_argument("--horizon-hours", nargs="+", type=int, default=[24, 168, 720])

    train_parser = subparsers.add_parser("train", help="Train models from resolved markets.")
    train_parser.add_argument("--artifact-dir", default="artifacts")
    train_parser.add_argument("--dataset", default=None)
    train_parser.add_argument("--max-pages", type=int, default=25)
    train_parser.add_argument("--page-size", type=int, default=200)
    train_parser.add_argument("--min-category-samples", type=int, default=40)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--model-type", default="logistic", choices=["logistic", "boosted_trees", "prior"])
    train_parser.add_argument("--edge-threshold", type=float, default=0.05)

    predict_parser = subparsers.add_parser("predict", help="Predict live market YES probabilities.")
    predict_parser.add_argument("--artifact-dir", default="artifacts")
    predict_parser.add_argument("--max-pages", type=int, default=5)
    predict_parser.add_argument("--page-size", type=int, default=200)
    predict_parser.add_argument("--limit", type=int, default=50)
    predict_parser.add_argument("--category", default=None)
    predict_parser.add_argument("--history-input", nargs="*", default=None)
    predict_parser.add_argument("--output", default=None, help="Optional CSV output path.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        path = fetch_snapshots(
            output_path=args.output,
            closed=args.closed,
            max_pages=args.max_pages,
            page_size=args.page_size,
            order=args.order,
            ascending=args.ascending,
            append=args.append,
        )
        print(str(path))
        return

    if args.command == "prepare":
        path = prepare_dataset(snapshot_paths=args.input, output_path=args.output)
        print(str(path))
        return

    if args.command == "backfill-markets":
        result = backfill_closed_markets(
            output_path=args.output,
            max_pages=args.max_pages,
            page_size=args.page_size,
            order=args.order,
            ascending=args.ascending,
            append=args.append,
        )
        print(json.dumps({"output": str(result.output_path), "records_written": result.records_written}, indent=2))
        return

    if args.command == "backfill-prices":
        result = backfill_price_history(
            market_snapshot_path=args.market_snapshots,
            output_path=args.output,
            interval=args.interval,
            fidelity=args.fidelity,
            max_markets=args.max_markets,
            append=args.append,
        )
        print(json.dumps({"output": str(result.output_path), "records_written": result.records_written}, indent=2))
        return

    if args.command == "build-horizon-dataset":
        result = build_horizon_dataset(
            price_history_path=args.price_history,
            output_path=args.output,
            horizon_hours=args.horizon_hours,
        )
        print(json.dumps({"output": str(result.output_path), "records_written": result.records_written}, indent=2))
        return

    if args.command == "train":
        result = train_models(
            artifact_dir=args.artifact_dir,
            dataset_path=args.dataset,
            max_pages=args.max_pages,
            page_size=args.page_size,
            min_category_samples=args.min_category_samples,
            validation_fraction=args.validation_fraction,
            model_type=args.model_type,
            edge_threshold=args.edge_threshold,
        )
        print(json.dumps(result.metrics, indent=2))
        return

    if args.command == "predict":
        frame = predict_open_markets(
            artifact_dir=args.artifact_dir,
            max_pages=args.max_pages,
            page_size=args.page_size,
            limit=args.limit,
            category_filter=args.category,
            history_snapshot_paths=args.history_input,
        )
        if args.output:
            frame.to_csv(args.output, index=False)
        if frame.empty:
            print("No matching open markets found.")
        else:
            print(frame.to_string(index=False))
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
