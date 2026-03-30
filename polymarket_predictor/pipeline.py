from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .client import GammaClient
from .features import FeatureRow, build_feature_row
from .model import LogisticModel, binary_accuracy, binary_log_loss


MIN_CATEGORY_SAMPLES = 40


@dataclass
class TrainingResult:
    bundle_path: Path
    metrics_path: Path
    metrics: dict[str, Any]


def collect_rows(markets: list[dict[str, Any]]) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    for market in markets:
        row = build_feature_row(market)
        if row is not None:
            rows.append(row)
    return rows


def to_frame(rows: list[FeatureRow]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "market_id": row.market_id,
                "slug": row.slug,
                "question": row.question,
                "category": row.category,
                "market_yes_probability": row.market_yes_probability,
                "label": row.label,
                "features": row.model_features,
            }
        )
    return pd.DataFrame.from_records(records)


def _stack_features(frame: pd.DataFrame) -> np.ndarray:
    return np.vstack(frame["features"].to_numpy())


def train_models(
    *,
    artifact_dir: str | Path,
    max_pages: int = 25,
    page_size: int = 200,
    min_category_samples: int = MIN_CATEGORY_SAMPLES,
) -> TrainingResult:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    client = GammaClient()
    historical_markets = client.fetch_market_pages(
        closed=True,
        max_pages=max_pages,
        page_size=page_size,
        order="updatedAt",
        ascending=False,
    )
    rows = collect_rows(historical_markets)
    frame = to_frame(rows)
    frame = frame.dropna(subset=["label"]).reset_index(drop=True)

    if frame.empty:
        raise RuntimeError("No resolved binary markets were available for training.")

    features = _stack_features(frame)
    labels = frame["label"].astype(int).to_numpy()

    global_model = LogisticModel.fit(features, labels)
    global_probabilities = global_model.predict_proba(features)

    models: dict[str, Any] = {"global": global_model.to_dict()}
    category_metrics: dict[str, Any] = {}

    category_counts = frame["category"].value_counts().to_dict()
    for category, count in category_counts.items():
        if count < min_category_samples:
            continue
        category_frame = frame[frame["category"] == category]
        category_features = _stack_features(category_frame)
        category_labels = category_frame["label"].astype(int).to_numpy()
        model = LogisticModel.fit(category_features, category_labels)
        probabilities = model.predict_proba(category_features)
        models[category] = model.to_dict()
        category_metrics[category] = {
            "samples": int(count),
            "accuracy": binary_accuracy(category_labels, probabilities),
            "log_loss": binary_log_loss(category_labels, probabilities),
            "base_rate": float(category_labels.mean()),
        }

    metrics = {
        "training_rows": int(len(frame)),
        "category_counts": {key: int(value) for key, value in category_counts.items()},
        "global": {
            "accuracy": binary_accuracy(labels, global_probabilities),
            "log_loss": binary_log_loss(labels, global_probabilities),
            "base_rate": float(labels.mean()),
        },
        "categories": category_metrics,
        "min_category_samples": min_category_samples,
    }

    bundle = {
        "models": models,
        "feature_count": int(features.shape[1]),
    }

    bundle_path = artifact_path / "model_bundle.json"
    metrics_path = artifact_path / "training_metrics.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainingResult(bundle_path=bundle_path, metrics_path=metrics_path, metrics=metrics)


def load_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    path = Path(artifact_dir) / "model_bundle.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing model bundle: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_open_markets(
    *,
    artifact_dir: str | Path,
    max_pages: int = 5,
    page_size: int = 200,
    limit: int | None = None,
    category_filter: str | None = None,
) -> pd.DataFrame:
    bundle = load_bundle(artifact_dir)
    models = {name: LogisticModel.from_dict(payload) for name, payload in bundle["models"].items()}
    global_model = models["global"]

    client = GammaClient()
    open_markets = client.fetch_market_pages(
        closed=False,
        max_pages=max_pages,
        page_size=page_size,
        order="volume24hr",
        ascending=False,
    )
    rows = collect_rows(open_markets)

    if category_filter:
        normalized_filter = category_filter.strip().lower()
        rows = [row for row in rows if row.category == normalized_filter]

    predictions: list[dict[str, Any]] = []
    for row in rows:
        features = row.model_features.reshape(1, -1)
        model = models.get(row.category, global_model)
        probability = float(model.predict_proba(features)[0])
        predictions.append(
            {
                "market_id": row.market_id,
                "slug": row.slug,
                "question": row.question,
                "category": row.category,
                "market_yes_probability": row.market_yes_probability,
                "predicted_yes_probability": probability,
                "edge": probability - row.market_yes_probability,
                "model_scope": row.category if row.category in models else "global",
            }
        )

    result = pd.DataFrame.from_records(predictions)
    if result.empty:
        return result
    result = result.sort_values(by=["edge", "predicted_yes_probability"], ascending=[False, False]).reset_index(drop=True)
    if limit is not None:
        result = result.head(limit).reset_index(drop=True)
    return result
