from __future__ import annotations

import argparse
import json

from .pipeline import predict_open_markets, train_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run category-aware Polymarket predictors.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train models from resolved markets.")
    train_parser.add_argument("--artifact-dir", default="artifacts")
    train_parser.add_argument("--max-pages", type=int, default=25)
    train_parser.add_argument("--page-size", type=int, default=200)
    train_parser.add_argument("--min-category-samples", type=int, default=40)

    predict_parser = subparsers.add_parser("predict", help="Predict live market YES probabilities.")
    predict_parser.add_argument("--artifact-dir", default="artifacts")
    predict_parser.add_argument("--max-pages", type=int, default=5)
    predict_parser.add_argument("--page-size", type=int, default=200)
    predict_parser.add_argument("--limit", type=int, default=50)
    predict_parser.add_argument("--category", default=None)
    predict_parser.add_argument("--output", default=None, help="Optional CSV output path.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        result = train_models(
            artifact_dir=args.artifact_dir,
            max_pages=args.max_pages,
            page_size=args.page_size,
            min_category_samples=args.min_category_samples,
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
