from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


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
