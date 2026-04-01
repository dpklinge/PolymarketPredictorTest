from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..ml.metrics import binary_accuracy, binary_log_loss, brier_score
from ..ml.pipeline import _blend_weight, _deserialize_feature_column, _split_chronologically, _stack_features, load_bundle


def load_training_metrics(artifact_dir: str | Path) -> dict[str, Any]:
    path = Path(artifact_dir) / "training_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing training metrics: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def training_comparison_frame(artifact_dirs: list[str | Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        metrics = load_training_metrics(artifact_dir)
        validation = metrics.get("global_validation", {})
        backtest = metrics.get("backtest_validation", {})
        records.append(
            {
                "artifact_dir": str(artifact_dir),
                "model_type": metrics.get("model_type", ""),
                "training_rows": metrics.get("training_rows", 0),
                "validation_rows": metrics.get("validation_rows", 0),
                "validation_accuracy": validation.get("accuracy"),
                "validation_log_loss": validation.get("log_loss"),
                "validation_brier": validation.get("brier_score"),
                "validation_ece": validation.get("ece"),
                "backtest_trades": backtest.get("trades"),
                "backtest_roi": backtest.get("roi"),
                "backtest_win_rate": backtest.get("win_rate"),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.sort_values(by=["validation_log_loss", "validation_brier"], ascending=[True, True]).reset_index(drop=True)


def prediction_comparison_frame(prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    normalized_frames: list[pd.DataFrame] = []
    for label, frame in prediction_frames.items():
        if frame.empty:
            continue
        subset = frame[
            [
                "market_id",
                "slug",
                "question",
                "category",
                "market_yes_probability",
                "predicted_yes_probability",
                "edge",
                "model_scope",
            ]
        ].copy()
        renamed = subset.rename(
            columns={
                "predicted_yes_probability": f"{label}_predicted_yes_probability",
                "edge": f"{label}_edge",
                "model_scope": f"{label}_model_scope",
            }
        )
        normalized_frames.append(renamed)

    if not normalized_frames:
        return pd.DataFrame()

    merged = normalized_frames[0]
    for frame in normalized_frames[1:]:
        merged = merged.merge(
            frame,
            on=["market_id", "slug", "question", "category", "market_yes_probability"],
            how="outer",
        )

    edge_columns = [column for column in merged.columns if column.endswith("_edge")]
    if edge_columns:
        merged["max_abs_edge"] = merged[edge_columns].abs().max(axis=1)
        merged = merged.sort_values(by=["max_abs_edge", "market_yes_probability"], ascending=[False, True])
    return merged.reset_index(drop=True)


def close_distance_efficacy_frame(artifact_dirs: list[str | Path], dataset_path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    dataset = _deserialize_feature_column(dataset)
    dataset = dataset.dropna(subset=["label"]).reset_index(drop=True)
    if dataset.empty:
        return pd.DataFrame()

    if "horizon_hours" in dataset.columns:
        dataset["lead_hours"] = pd.to_numeric(dataset["horizon_hours"], errors="coerce")
        labeler = lambda series: series.round().astype(int).astype(str) + "h"
    elif {"resolved_at", "fetched_at"}.issubset(dataset.columns):
        resolved = pd.to_datetime(dataset["resolved_at"], utc=True, errors="coerce")
        fetched = pd.to_datetime(dataset["fetched_at"], utc=True, errors="coerce")
        dataset["lead_hours"] = (resolved - fetched).dt.total_seconds() / 3600.0
        labeler = _continuous_distance_labels
    else:
        raise ValueError("The selected dataset does not contain horizon or fetch timing information needed for a close-distance graph.")

    records: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        metrics_path = Path(artifact_dir) / "training_metrics.json"
        validation_fraction = 0.2
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            validation_fraction = float(metrics.get("split", {}).get("validation_fraction", 0.2))

        train_frame, validation_frame = _split_chronologically(dataset, validation_fraction)
        if validation_frame.empty:
            continue

        bundle = load_bundle(artifact_dir)
        models = bundle["models"]
        global_model = models["global"]
        category_train_rows = {key: int(value) for key, value in bundle.get("category_train_rows", {}).items()}
        min_category_samples = int(bundle.get("min_category_samples", 40))

        validation_features = _stack_features(validation_frame)
        global_probabilities = global_model.predict_proba(validation_features)
        blended_probabilities = global_probabilities.copy()

        for category, train_rows in category_train_rows.items():
            category_validation = validation_frame[validation_frame["category"] == category]
            if category_validation.empty or category not in models:
                continue
            indexes = category_validation.index.to_numpy()
            category_features = _stack_features(category_validation)
            category_probabilities = models[category].predict_proba(category_features)
            blend_weight = _blend_weight(train_rows, min_category_samples)
            blended_probabilities[indexes] = (
                blend_weight * category_probabilities
                + (1.0 - blend_weight) * global_probabilities[indexes]
            )

        scored = validation_frame.copy()
        scored["predicted_yes_probability"] = blended_probabilities
        scored = scored.dropna(subset=["lead_hours"]).reset_index(drop=True)
        if scored.empty:
            continue

        if "horizon_hours" in scored.columns:
            scored["distance_label"] = labeler(scored["lead_hours"])
            grouped_items = sorted(scored.groupby("distance_label"), key=lambda item: float(item[1]["lead_hours"].mean()))
        else:
            scored["distance_label"] = labeler(scored["lead_hours"])
            grouped_items = sorted(scored.groupby("distance_label"), key=lambda item: float(item[1]["lead_hours"].mean()))

        run_label = Path(artifact_dir).name or str(artifact_dir)
        for distance_label, group in grouped_items:
            labels = group["label"].astype(int).to_numpy()
            probabilities = group["predicted_yes_probability"].astype(float).to_numpy()
            lead_mid = float(group["lead_hours"].mean())
            records.append(
                {
                    "artifact_dir": str(artifact_dir),
                    "run_label": run_label,
                    "distance_label": distance_label,
                    "lead_hours_midpoint": lead_mid,
                    "rows": int(len(group)),
                    "accuracy": binary_accuracy(labels, probabilities),
                    "log_loss": binary_log_loss(labels, probabilities),
                    "brier_score": brier_score(labels, probabilities),
                }
            )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.sort_values(by=["lead_hours_midpoint", "run_label"]).reset_index(drop=True)


def _continuous_distance_labels(lead_hours: pd.Series) -> pd.Series:
    edges = [-np.inf, 24, 72, 168, 336, 720, 1440, np.inf]
    labels = ["<24h", "1-3d", "3-7d", "1-2w", "2-4w", "1-2mo", "2mo+"]
    return pd.cut(lead_hours, bins=edges, labels=labels, include_lowest=True).astype(str)


def snapshot_review_summary_frame(review_frame: pd.DataFrame) -> pd.DataFrame:
    if review_frame.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for model_label, group in review_frame.groupby("model_label"):
        resolved = group[group["verdict"].isin(["Success", "Failure"])].copy()
        pending = group[group["verdict"] == "Pending"].copy()
        wins = int((resolved["verdict"] == "Success").sum())
        losses = int((resolved["verdict"] == "Failure").sum())
        resolved_count = int(len(resolved))
        pending_count = int(len(pending))
        total_count = int(len(group))
        total_cost = float(group["stake_cost_at_snapshot"].fillna(0.0).sum())
        resolved_cost = float(resolved["stake_cost_at_snapshot"].fillna(0.0).sum())
        resolved_payout = float(resolved["realized_payout"].fillna(0.0).sum())
        resolved_pnl = float(resolved["realized_pnl"].fillna(0.0).sum())
        pending_max_profit = float(pending["max_profit_at_snapshot"].fillna(0.0).sum())
        pending_max_loss = float(pending["max_loss_at_snapshot"].fillna(0.0).sum())

        budget_per_pick = 1000.0 / total_count if total_count > 0 else 0.0
        resolved_with_cost = resolved[resolved["stake_cost_at_snapshot"].fillna(0.0) > 0]
        if len(resolved_with_cost) > 0:
            dollar_pnl_per_pick = (
                resolved_with_cost["realized_pnl"].fillna(0.0)
                / resolved_with_cost["stake_cost_at_snapshot"]
                * budget_per_pick
            )
            expected_return_1000 = float(dollar_pnl_per_pick.sum())
        else:
            expected_return_1000 = None

        records.append(
            {
                "model_label": model_label,
                "rows": total_count,
                "resolved_rows": resolved_count,
                "pending_rows": pending_count,
                "wins": wins,
                "losses": losses,
                "win_rate": (wins / resolved_count) if resolved_count else None,
                "total_cost_at_snapshot": total_cost,
                "resolved_cost_at_snapshot": resolved_cost,
                "resolved_payout": resolved_payout,
                "resolved_pnl": resolved_pnl,
                "avg_resolved_pnl": (resolved_pnl / resolved_count) if resolved_count else None,
                "resolved_roi": (resolved_pnl / resolved_cost) if resolved_cost else None,
                "pending_max_profit": pending_max_profit,
                "pending_max_loss": pending_max_loss,
                "expected_return_1000": expected_return_1000,
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.sort_values(by=["resolved_pnl", "win_rate"], ascending=[False, False]).reset_index(drop=True)
