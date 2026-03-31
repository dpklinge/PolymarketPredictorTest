from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from polymarket_predictor.datasets.backfill import build_horizon_dataset
from polymarket_predictor.datasets.data import (
    build_market_history_index,
    load_snapshot_file,
    prepare_dataset,
    temporal_features_from_history,
)
from polymarket_predictor.datasets.features import TEMPORAL_DIMENSIONS, TEXT_DIMENSIONS, build_feature_row
from polymarket_predictor.ml.metrics import simulate_backtest
from polymarket_predictor.ml.model import GradientBoostedStumpModel, LogisticModel
from polymarket_predictor.ml.models import create_model, load_model
from polymarket_predictor.ml.pipeline import _blend_weight, _split_chronologically
from polymarket_predictor.review.snapshotting import (
    build_snapshot_output_path,
    compare_prediction_snapshots,
    save_prediction_snapshots,
)
from polymarket_predictor.ui.gui_utils import prediction_comparison_frame, training_comparison_frame


def synthetic_market(index: int, *, category: str, yes_price: float, resolved_yes: bool, closed: bool) -> dict[str, object]:
    final_prices = [1.0, 0.0] if resolved_yes else [0.0, 1.0]
    prices = final_prices if closed else [yes_price, 1.0 - yes_price]
    return {
        "id": str(index),
        "slug": f"market-{index}",
        "question": f"Will event {index} happen?",
        "description": f"{category} market {index}",
        "category": category,
        "outcomePrices": json.dumps(prices),
        "volumeNum": 1000 + index,
        "liquidityNum": 500 + index,
        "active": not closed,
        "closed": closed,
        "enableOrderBook": True,
    }


def test_feature_row_extracts_binary_probability() -> None:
    row = build_feature_row(synthetic_market(1, category="politics", yes_price=0.62, resolved_yes=False, closed=False))
    assert row is not None
    assert row.category == "politics"
    assert abs(row.market_yes_probability - 0.62) < 1e-9
    assert row.label is None


def test_logistic_model_learns_signal() -> None:
    x = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.1],
            [0.8, 0.9],
            [0.9, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1], dtype=int)

    model = LogisticModel.fit(x, y, epochs=1200)
    probabilities = model.predict_proba(x)

    assert probabilities[0] < 0.5
    assert probabilities[-1] > 0.5


def test_chronological_split_orders_by_time() -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["1", "2", "3", "4"],
            "resolved_at": [
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
                "2026-01-03T00:00:00+00:00",
                "2026-01-04T00:00:00+00:00",
            ],
            "features": [np.array([0.1]), np.array([0.2]), np.array([0.3]), np.array([0.4])],
            "label": [0, 0, 1, 1],
            "category": ["a", "a", "a", "a"],
        }
    )
    train_frame, validation_frame = _split_chronologically(frame, 0.25)
    assert list(train_frame["market_id"]) == ["1", "2", "3"]
    assert list(validation_frame["market_id"]) == ["4"]


def test_backtest_takes_yes_and_no_trades() -> None:
    frame = pd.DataFrame(
        {
            "predicted_yes_probability": [0.8, 0.2, 0.52],
            "market_yes_probability": [0.6, 0.4, 0.5],
            "label": [1, 0, 1],
        }
    )
    summary = simulate_backtest(frame, edge_threshold=0.1)
    assert summary["trades"] == 2
    assert summary["roi"] > 0.0
    assert summary["yes_trades"] == 1
    assert summary["no_trades"] == 1


def test_prepare_dataset_adds_temporal_history_features(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshots.jsonl"
    market_one = synthetic_market(1, category="sports", yes_price=0.40, resolved_yes=False, closed=False)
    market_two = synthetic_market(1, category="sports", yes_price=0.55, resolved_yes=True, closed=True)
    lines = [
        json.dumps({"fetched_at": "2026-03-01T00:00:00+00:00", "market": market_one}),
        json.dumps({"fetched_at": "2026-03-02T00:00:00+00:00", "market": market_two}),
    ]
    snapshot_path.write_text("\n".join(lines), encoding="utf-8")

    output_path = tmp_path / "prepared.csv"
    prepare_dataset(snapshot_paths=[snapshot_path], output_path=output_path)
    frame = pd.read_csv(output_path)

    assert len(frame) == 2
    first_features = np.asarray(json.loads(frame.iloc[0]["features"]), dtype=float)
    second_features = np.asarray(json.loads(frame.iloc[1]["features"]), dtype=float)
    temporal_start = len(second_features) - TEXT_DIMENSIONS - TEMPORAL_DIMENSIONS
    assert len(second_features) == len(first_features)
    assert second_features[temporal_start + 7] == 1.0
    assert abs(second_features[temporal_start + 1]) > 0.0


def test_blend_weight_increases_with_more_category_rows() -> None:
    assert _blend_weight(10, 40) < _blend_weight(100, 40)


def test_build_market_history_index_uses_latest_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "history.jsonl"
    lines = [
        json.dumps({"fetched_at": "2026-03-01T00:00:00+00:00", "market": synthetic_market(1, category="sports", yes_price=0.40, resolved_yes=False, closed=False)}),
        json.dumps({"fetched_at": "2026-03-02T00:00:00+00:00", "market": synthetic_market(1, category="sports", yes_price=0.55, resolved_yes=False, closed=False)}),
    ]
    snapshot_path.write_text("\n".join(lines), encoding="utf-8")

    history_index = build_market_history_index([snapshot_path])
    previous_market, previous_fetched_at = history_index["1"]
    assert previous_fetched_at == "2026-03-02T00:00:00+00:00"
    assert json.loads(previous_market["outcomePrices"])[0] == 0.55


def test_temporal_features_from_history_uses_previous_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "history.jsonl"
    earlier_market = synthetic_market(1, category="sports", yes_price=0.40, resolved_yes=False, closed=False)
    later_market = synthetic_market(1, category="sports", yes_price=0.55, resolved_yes=False, closed=False)
    snapshot_path.write_text(
        json.dumps({"fetched_at": "2026-03-01T00:00:00+00:00", "market": earlier_market}),
        encoding="utf-8",
    )
    history_index = build_market_history_index([snapshot_path])
    temporal = temporal_features_from_history(
        later_market,
        fetched_at="2026-03-02T00:00:00+00:00",
        history_index=history_index,
    )
    assert temporal[-1] == 1.0
    assert temporal[1] > 0.0


def test_boosted_stump_model_learns_nonlinear_signal() -> None:
    x = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.2],
            [0.8, 0.1],
            [0.9, 0.9],
            [1.0, 0.8],
            [0.0, 0.9],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 1, 0], dtype=int)

    model = GradientBoostedStumpModel.fit(x, y, n_estimators=25)
    probabilities = model.predict_proba(x)

    assert probabilities[0] < 0.5
    assert probabilities[3] > 0.5


def test_boosted_tree_adapter_round_trips() -> None:
    x = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)

    adapter = create_model("boosted_trees").fit(x, y)
    payload = adapter.to_dict()
    restored = load_model(payload)
    probabilities = restored.predict_proba(x)

    assert payload["model_type"] == "boosted_trees"
    assert probabilities[-1] > probabilities[0]


def test_logistic_adapter_calibration_round_trips() -> None:
    x = np.array([[0.1], [0.2], [0.8], [0.9], [0.85], [0.15], [0.7], [0.25]], dtype=float)
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=int)

    adapter = create_model("logistic").fit(x[:6], y[:6], calibration_features=x[6:], calibration_labels=y[6:])
    payload = adapter.to_dict()
    restored = load_model(payload)
    probabilities = restored.predict_proba(x)

    assert "calibrator" in payload
    assert np.all(probabilities > 0.0)
    assert np.all(probabilities < 1.0)


def test_training_comparison_frame_sorts_by_log_loss(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "training_metrics.json").write_text(
        json.dumps(
            {
                "model_type": "logistic",
                "training_rows": 10,
                "validation_rows": 5,
                "global_validation": {"accuracy": 0.8, "log_loss": 0.2, "brier_score": 0.1, "ece": 0.05},
                "backtest_validation": {"trades": 2, "roi": 0.1, "win_rate": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (run_b / "training_metrics.json").write_text(
        json.dumps(
            {
                "model_type": "boosted_trees",
                "training_rows": 10,
                "validation_rows": 5,
                "global_validation": {"accuracy": 0.9, "log_loss": 0.1, "brier_score": 0.05, "ece": 0.03},
                "backtest_validation": {"trades": 3, "roi": 0.2, "win_rate": 0.7},
            }
        ),
        encoding="utf-8",
    )

    frame = training_comparison_frame([run_a, run_b])
    assert list(frame["model_type"]) == ["boosted_trees", "logistic"]


def test_prediction_comparison_frame_merges_predictions() -> None:
    logistic = pd.DataFrame(
        {
            "market_id": ["1"],
            "slug": ["market-1"],
            "question": ["Will x happen?"],
            "category": ["sports"],
            "market_yes_probability": [0.4],
            "predicted_yes_probability": [0.55],
            "edge": [0.15],
            "model_scope": ["global"],
        }
    )
    boosted = pd.DataFrame(
        {
            "market_id": ["1"],
            "slug": ["market-1"],
            "question": ["Will x happen?"],
            "category": ["sports"],
            "market_yes_probability": [0.4],
            "predicted_yes_probability": [0.62],
            "edge": [0.22],
            "model_scope": ["global"],
        }
    )

    frame = prediction_comparison_frame({"logistic": logistic, "boosted": boosted})
    assert "logistic_edge" in frame.columns
    assert "boosted_predicted_yes_probability" in frame.columns
    assert frame.iloc[0]["market_id"] == "1"


def test_build_horizon_dataset_creates_anchor_rows(tmp_path: Path) -> None:
    history_path = tmp_path / "prices.jsonl"
    market = synthetic_market(1, category="sports", yes_price=0.8, resolved_yes=True, closed=True)
    market["conditionId"] = "condition-1"
    market["startDate"] = "2026-03-01T00:00:00+00:00"
    market["endDate"] = "2026-03-10T00:00:00+00:00"
    history_payload = {
        "market_id": "1",
        "condition_id": "condition-1",
        "interval": "1d",
        "fidelity": 60,
        "market": market,
        "history": [
            {"t": 1772323200, "p": 0.25},
            {"t": 1772496000, "p": 0.35},
            {"t": 1772582400, "p": 0.45},
            {"t": 1772668800, "p": 0.55},
        ],
    }
    history_path.write_text(json.dumps(history_payload), encoding="utf-8")

    output_path = tmp_path / "horizon.csv"
    result = build_horizon_dataset(price_history_path=history_path, output_path=output_path, horizon_hours=[24, 48])
    frame = pd.read_csv(output_path)

    assert result.records_written == 2
    assert sorted(frame["horizon_hours"].tolist()) == [24, 48]
    assert all(frame["label"] == 1)


def test_prepare_dataset_removes_duplicate_snapshot_rows(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "duplicate_snapshots.jsonl"
    market = synthetic_market(1, category="sports", yes_price=0.4, resolved_yes=False, closed=False)
    payload = json.dumps({"fetched_at": "2026-03-01T00:00:00+00:00", "market": market})
    snapshot_path.write_text("\n".join([payload, payload]), encoding="utf-8")

    output_path = tmp_path / "prepared.csv"
    prepare_dataset(snapshot_paths=[snapshot_path], output_path=output_path)
    frame = pd.read_csv(output_path)

    assert len(frame) == 1


def test_build_horizon_dataset_dedupes_duplicate_price_payloads(tmp_path: Path) -> None:
    history_path = tmp_path / "duplicate_prices.jsonl"
    market = synthetic_market(1, category="sports", yes_price=0.8, resolved_yes=True, closed=True)
    market["conditionId"] = "condition-1"
    market["startDate"] = "2026-03-01T00:00:00+00:00"
    market["endDate"] = "2026-03-10T00:00:00+00:00"
    payload = {
        "market_id": "1",
        "condition_id": "condition-1",
        "interval": "1d",
        "fidelity": 60,
        "market": market,
        "history": [
            {"t": 1772323200, "p": 0.25},
            {"t": 1772496000, "p": 0.35},
            {"t": 1772582400, "p": 0.45},
            {"t": 1772668800, "p": 0.55},
        ],
    }
    serialized = json.dumps(payload)
    history_path.write_text("\n".join([serialized, serialized]), encoding="utf-8")

    output_path = tmp_path / "horizon.csv"
    result = build_horizon_dataset(price_history_path=history_path, output_path=output_path, horizon_hours=[24])
    frame = pd.read_csv(output_path)

    assert result.records_written == 1
    assert len(frame) == 1


def test_save_prediction_snapshots_writes_rows(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "market_id": ["1"],
            "slug": ["market-1"],
            "question": ["Will x happen?"],
            "category": ["sports"],
            "market_yes_probability": [0.4],
            "predicted_yes_probability": [0.7],
            "edge": [0.3],
            "model_scope": ["global"],
        }
    )
    output_path = tmp_path / "snapshots"
    result = save_prediction_snapshots({"logistic": predictions}, output_path=output_path, append=False)
    frame = pd.read_csv(result.output_path)

    assert result.records_written == 1
    assert len(frame) == 1
    assert frame.iloc[0]["model_label"] == "logistic"
    assert result.output_path.name.startswith("predictions_")
    assert result.output_path.parent.name.count("-") == 2


def test_build_snapshot_output_path_uses_date_folder_and_timestamped_name(tmp_path: Path) -> None:
    path = build_snapshot_output_path(tmp_path / "prediction_snapshots", snapshot_time="2026-03-31T12:34:56+00:00")
    assert path.parent.name == "2026-03-31"
    assert path.name == "predictions_2026-03-31_12-34-56.csv"


def test_compare_prediction_snapshots_marks_success_and_pending(tmp_path: Path, monkeypatch) -> None:
    snapshot_path = tmp_path / "snapshots.csv"
    pd.DataFrame(
        {
            "snapshot_time": ["2026-03-31T12:00:00+00:00", "2026-03-31T12:00:00+00:00"],
            "model_label": ["logistic", "logistic"],
            "artifact_dir": ["artifacts/logistic_run", "artifacts/logistic_run"],
            "market_id": ["1", "2"],
            "slug": ["market-1", "market-2"],
            "question": ["Will x happen?", "Will y happen?"],
            "category": ["sports", "politics"],
            "market_yes_probability_at_snapshot": [0.4, 0.6],
            "predicted_yes_probability": [0.8, 0.2],
            "edge_at_snapshot": [0.4, -0.4],
            "model_scope": ["global", "global"],
        }
    ).to_csv(snapshot_path, index=False)

    class FakeClient:
        def get_market(self, market_id: str):
            if market_id == "1":
                return {"outcomePrices": json.dumps([1.0, 0.0]), "closed": True}
            return {"outcomePrices": json.dumps([0.55, 0.45]), "closed": False}

    monkeypatch.setattr("polymarket_predictor.review.snapshotting.GammaClient", FakeClient)
    frame = compare_prediction_snapshots(snapshot_path, limit=10, status_filter="All")

    assert frame.columns[0] == "verdict"
    verdicts = dict(zip(frame["market_id"], frame["verdict"]))
    assert verdicts["1"] == "Success"
    assert verdicts["2"] == "Pending"
