from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..api.client import GammaClient
from ..datasets.data import utc_now_iso
from ..datasets.features import extract_yes_probability, infer_resolution_label


@dataclass
class SnapshotSaveResult:
    output_path: Path
    records_written: int


def save_prediction_snapshots(
    prediction_frames: dict[str, pd.DataFrame],
    *,
    output_path: str | Path,
    append: bool = True,
    artifact_dir_lookup: dict[str, str] | None = None,
) -> SnapshotSaveResult:
    snapshot_time = utc_now_iso()
    records: list[dict[str, Any]] = []
    for model_label, frame in prediction_frames.items():
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            records.append(
                {
                    "snapshot_time": snapshot_time,
                    "model_label": model_label,
                    "artifact_dir": (artifact_dir_lookup or {}).get(model_label, ""),
                    "market_id": getattr(row, "market_id"),
                    "slug": getattr(row, "slug"),
                    "question": getattr(row, "question"),
                    "category": getattr(row, "category"),
                    "market_yes_probability_at_snapshot": getattr(row, "market_yes_probability"),
                    "predicted_yes_probability": getattr(row, "predicted_yes_probability"),
                    "edge_at_snapshot": getattr(row, "edge"),
                    "model_scope": getattr(row, "model_scope"),
                }
            )

    destination = build_snapshot_output_path(output_path, snapshot_time=snapshot_time)
    destination.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame.from_records(records)
    if new_frame.empty:
        raise RuntimeError("No prediction rows were available to snapshot.")

    if append and destination.exists():
        existing = pd.read_csv(destination)
        combined = pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = new_frame
    combined = combined.drop_duplicates(subset=["snapshot_time", "model_label", "market_id"]).reset_index(drop=True)
    combined.to_csv(destination, index=False)
    return SnapshotSaveResult(output_path=destination, records_written=len(new_frame))


def build_snapshot_output_path(output_path: str | Path, *, snapshot_time: str) -> Path:
    requested = Path(output_path)
    timestamp = _parse_snapshot_time(snapshot_time)
    date_folder = timestamp.strftime("%Y-%m-%d")
    filename = f"predictions_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    if requested.suffix.lower() == ".csv":
        base_dir = requested.parent
    else:
        base_dir = requested
    return base_dir / date_folder / filename


def compare_prediction_snapshots(
    snapshot_path: str | Path,
    *,
    limit: int | None = None,
    status_filter: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(snapshot_path)
    if frame.empty:
        return frame

    client = GammaClient()
    current_markets: dict[str, dict[str, Any]] = {}
    for market_id in frame["market_id"].astype(str).unique():
        current_markets[market_id] = client.get_market(market_id)

    comparison_records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        market_id = str(getattr(row, "market_id"))
        current_market = current_markets[market_id]
        current_yes_probability = extract_yes_probability(current_market)
        resolution_label = infer_resolution_label(current_market)
        predicted_yes_probability = float(getattr(row, "predicted_yes_probability"))
        predicted_side = "YES" if predicted_yes_probability >= 0.5 else "NO"

        if resolution_label is None:
            verdict = "Pending"
            actual_side = ""
        else:
            actual_side = "YES" if resolution_label == 1 else "NO"
            verdict = "Success" if predicted_side == actual_side else "Failure"

        comparison_records.append(
            {
                "snapshot_time": getattr(row, "snapshot_time"),
                "model_label": getattr(row, "model_label"),
                "artifact_dir": getattr(row, "artifact_dir"),
                "market_id": market_id,
                "slug": getattr(row, "slug"),
                "question": getattr(row, "question"),
                "category": getattr(row, "category"),
                "predicted_side": predicted_side,
                "predicted_yes_probability": predicted_yes_probability,
                "market_yes_probability_at_snapshot": float(getattr(row, "market_yes_probability_at_snapshot")),
                "current_market_yes_probability": float(current_yes_probability) if current_yes_probability is not None else None,
                "current_closed": bool(current_market.get("closed")),
                "actual_side": actual_side,
                "verdict": verdict,
                "current_edge_vs_snapshot_market": (
                    float(current_yes_probability) - float(getattr(row, "market_yes_probability_at_snapshot"))
                    if current_yes_probability is not None
                    else None
                ),
            }
        )

    comparison = pd.DataFrame.from_records(comparison_records)
    if status_filter and status_filter != "All":
        comparison = comparison[comparison["verdict"] == status_filter].reset_index(drop=True)
    status_rank = {"Failure": 0, "Pending": 1, "Success": 2}
    comparison["_status_rank"] = comparison["verdict"].map(status_rank).fillna(9)
    comparison = comparison.sort_values(
        by=["_status_rank", "snapshot_time", "predicted_yes_probability"],
        ascending=[True, False, False],
    ).drop(columns="_status_rank")
    ordered_columns = [
        "verdict",
        "snapshot_time",
        "model_label",
        "artifact_dir",
        "market_id",
        "slug",
        "question",
        "category",
        "predicted_side",
        "actual_side",
        "predicted_yes_probability",
        "market_yes_probability_at_snapshot",
        "current_market_yes_probability",
        "current_closed",
        "current_edge_vs_snapshot_market",
    ]
    comparison = comparison[[column for column in ordered_columns if column in comparison.columns]]
    if limit is not None:
        comparison = comparison.head(limit).reset_index(drop=True)
    return comparison.reset_index(drop=True)


def _parse_snapshot_time(snapshot_time: str) -> datetime:
    normalized = snapshot_time.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)
