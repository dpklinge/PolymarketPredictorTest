from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..api.client import GammaClient
from ..datasets.data import build_market_history_index, prepare_dataset, temporal_features_from_history, utc_now_iso
from ..datasets.features import FeatureRow, build_feature_row
from ..datasets.taxonomy import load_taxonomy
from .metrics import simulate_backtest, summarize_classification
from .models import create_model, load_model


MIN_CATEGORY_SAMPLES = 40


@dataclass
class TrainingResult:
    bundle_path: Path
    metrics_path: Path
    metrics: dict[str, Any]


def collect_rows(markets: list[dict[str, Any]]) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    taxonomy = load_taxonomy()
    for market in markets:
        row = build_feature_row(market, taxonomy=taxonomy)
        if row is not None:
            rows.append(row)
    return rows


def collect_rows_with_history(
    markets: list[dict[str, Any]],
    *,
    history_index: dict[str, tuple[dict[str, Any], str | None]] | None = None,
    fetched_at: str | None = None,
) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    taxonomy = load_taxonomy()
    inferred_fetched_at = fetched_at or utc_now_iso()
    for market in markets:
        temporal_features = temporal_features_from_history(
            market,
            fetched_at=inferred_fetched_at,
            history_index=history_index,
        )
        row = build_feature_row(market, taxonomy=taxonomy, temporal_features=temporal_features)
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


def _blend_weight(train_rows: int, min_category_samples: int) -> float:
    return float(min(0.85, train_rows / max(train_rows + min_category_samples, 1)))


def _deserialize_feature_column(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    copy["features"] = copy["features"].apply(
        lambda value: np.asarray(json.loads(value), dtype=float) if isinstance(value, str) else value
    )
    return copy


def _load_training_frame(
    *,
    dataset_path: str | Path | None,
    max_pages: int,
    page_size: int,
) -> pd.DataFrame:
    if dataset_path:
        frame = pd.read_csv(dataset_path)
        frame = _deserialize_feature_column(frame)
        return frame.dropna(subset=["label"]).reset_index(drop=True)

    artifact_dataset = Path("artifacts") / "prepared_training_data.csv"
    snapshot_path = Path("artifacts") / "raw_closed_snapshots.jsonl"
    client = GammaClient()
    historical_markets = client.fetch_market_pages(
        closed=True,
        max_pages=max_pages,
        page_size=page_size,
        order="updatedAt",
        ascending=False,
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_at = pd.Timestamp.utcnow().isoformat()
    snapshot_path.write_text(
        "".join(json.dumps({"fetched_at": fetched_at, "market": market}) + "\n" for market in historical_markets),
        encoding="utf-8",
    )
    prepare_dataset(snapshot_paths=[snapshot_path], output_path=artifact_dataset)
    frame = pd.read_csv(artifact_dataset)
    frame = _deserialize_feature_column(frame)
    return frame.dropna(subset=["label"]).reset_index(drop=True)


def _split_chronologically(frame: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sortable = frame.copy()
    sortable["resolved_at"] = pd.to_datetime(sortable["resolved_at"], utc=True, errors="coerce")
    sortable = sortable.sort_values(by=["resolved_at", "market_id"], ascending=[True, True]).reset_index(drop=True)
    split_index = max(1, int(len(sortable) * (1.0 - validation_fraction)))
    split_index = min(split_index, len(sortable) - 1)
    return sortable.iloc[:split_index].reset_index(drop=True), sortable.iloc[split_index:].reset_index(drop=True)


def train_models(
    *,
    artifact_dir: str | Path,
    dataset_path: str | Path | None = None,
    max_pages: int = 25,
    page_size: int = 200,
    min_category_samples: int = MIN_CATEGORY_SAMPLES,
    validation_fraction: float = 0.2,
    model_type: str = "logistic",
    edge_threshold: float = 0.05,
) -> TrainingResult:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    frame = _load_training_frame(dataset_path=dataset_path, max_pages=max_pages, page_size=page_size)

    if frame.empty:
        raise RuntimeError("No resolved binary markets were available for training.")

    train_frame, validation_frame = _split_chronologically(frame, validation_fraction)
    train_features = _stack_features(train_frame)
    train_labels = train_frame["label"].astype(int).to_numpy()
    validation_features = _stack_features(validation_frame)
    validation_labels = validation_frame["label"].astype(int).to_numpy()

    global_model = create_model(model_type).fit(
        train_features,
        train_labels,
        calibration_features=validation_features,
        calibration_labels=validation_labels,
    )
    train_global_probabilities = global_model.predict_proba(train_features)
    validation_global_probabilities = global_model.predict_proba(validation_features)

    models: dict[str, Any] = {"global": global_model.to_dict()}
    category_train_rows: dict[str, int] = {}
    category_metrics: dict[str, Any] = {}

    category_counts = frame["category"].value_counts().to_dict()
    for category, count in category_counts.items():
        category_train = train_frame[train_frame["category"] == category]
        if count < min_category_samples or len(category_train) < min_category_samples:
            continue
        category_features = _stack_features(category_train)
        category_labels = category_train["label"].astype(int).to_numpy()
        category_validation = validation_frame[validation_frame["category"] == category]
        category_validation_features = _stack_features(category_validation) if not category_validation.empty else None
        category_validation_labels = category_validation["label"].astype(int).to_numpy() if not category_validation.empty else None
        model = create_model(model_type).fit(
            category_features,
            category_labels,
            calibration_features=category_validation_features,
            calibration_labels=category_validation_labels,
        )
        models[category] = model.to_dict()
        category_train_rows[category] = int(len(category_train))
        if category_validation.empty:
            continue
        category_probabilities = model.predict_proba(category_validation_features)
        global_probabilities = global_model.predict_proba(category_validation_features)
        blend_weight = _blend_weight(len(category_train), min_category_samples)
        probabilities = blend_weight * category_probabilities + (1.0 - blend_weight) * global_probabilities
        category_metrics[category] = {
            "samples": int(count),
            "train_rows": int(len(category_train)),
            "blend_weight": blend_weight,
            "validation": summarize_classification(category_validation_labels, probabilities),
        }

    validation_scored = validation_frame.copy()
    validation_scored["predicted_yes_probability"] = validation_global_probabilities
    blended_validation_probabilities = validation_global_probabilities.copy()
    for category, train_rows in category_train_rows.items():
        category_validation = validation_frame[validation_frame["category"] == category]
        if category_validation.empty:
            continue
        category_indexes = category_validation.index.to_numpy()
        category_features = _stack_features(category_validation)
        category_probabilities = load_model(models[category]).predict_proba(category_features)
        blend_weight = _blend_weight(train_rows, min_category_samples)
        blended_validation_probabilities[category_indexes] = (
            blend_weight * category_probabilities
            + (1.0 - blend_weight) * validation_global_probabilities[category_indexes]
        )
    validation_scored["predicted_yes_probability"] = blended_validation_probabilities

    metrics = {
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "category_counts": {key: int(value) for key, value in category_counts.items()},
        "global_train": summarize_classification(train_labels, train_global_probabilities),
        "global_validation": summarize_classification(validation_labels, validation_global_probabilities),
        "backtest_validation": simulate_backtest(validation_scored, edge_threshold=edge_threshold),
        "split": {
            "validation_fraction": validation_fraction,
            "train_start": str(train_frame["resolved_at"].min()),
            "train_end": str(train_frame["resolved_at"].max()),
            "validation_start": str(validation_frame["resolved_at"].min()),
            "validation_end": str(validation_frame["resolved_at"].max()),
        },
        "categories": category_metrics,
        "min_category_samples": min_category_samples,
        "model_type": model_type,
        "edge_threshold": edge_threshold,
    }

    bundle = {
        "models": models,
        "category_train_rows": category_train_rows,
        "feature_count": int(train_features.shape[1]),
        "model_type": model_type,
        "min_category_samples": min_category_samples,
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
    history_snapshot_paths: list[str | Path] | None = None,
) -> pd.DataFrame:
    bundle = load_bundle(artifact_dir)
    models = {name: load_model(payload) for name, payload in bundle["models"].items()}
    global_model = models["global"]
    category_train_rows = {key: int(value) for key, value in bundle.get("category_train_rows", {}).items()}
    min_category_samples = int(bundle.get("min_category_samples", MIN_CATEGORY_SAMPLES))

    client = GammaClient()
    open_markets = client.fetch_market_pages(
        closed=False,
        max_pages=max_pages,
        page_size=page_size,
        order="volume24hr",
        ascending=False,
    )
    fetched_at = utc_now_iso()
    history_index = build_market_history_index(history_snapshot_paths or []) if history_snapshot_paths else None
    rows = collect_rows_with_history(open_markets, history_index=history_index, fetched_at=fetched_at)

    if category_filter:
        normalized_filter = category_filter.strip().lower()
        rows = [row for row in rows if row.category == normalized_filter]

    predictions: list[dict[str, Any]] = []
    for row in rows:
        features = row.model_features.reshape(1, -1)
        global_probability = float(global_model.predict_proba(features)[0])
        category_model = models.get(row.category)
        if category_model is not None and row.category in category_train_rows:
            category_probability = float(category_model.predict_proba(features)[0])
            blend_weight = _blend_weight(category_train_rows[row.category], min_category_samples)
            probability = blend_weight * category_probability + (1.0 - blend_weight) * global_probability
            model_scope = f"blend:{row.category}"
        else:
            probability = global_probability
            model_scope = "global"
        predictions.append(
            {
                "market_id": row.market_id,
                "slug": row.slug,
                "question": row.question,
                "category": row.category,
                "market_yes_probability": row.market_yes_probability,
                "predicted_yes_probability": probability,
                "edge": probability - row.market_yes_probability,
                "model_scope": model_scope,
            }
        )

    result = pd.DataFrame.from_records(predictions)
    if result.empty:
        return result
    result = result.sort_values(by=["edge", "predicted_yes_probability"], ascending=[False, False]).reset_index(drop=True)
    if limit is not None:
        result = result.head(limit).reset_index(drop=True)
    return result
