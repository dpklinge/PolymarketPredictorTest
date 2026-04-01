from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return float(-(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)).mean())


def binary_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = (probabilities >= 0.5).astype(int)
    return float((predictions == labels).mean())


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= 1.0:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        if not np.any(mask):
            continue
        bucket_prob = probabilities[mask].mean()
        bucket_label = labels[mask].mean()
        total += abs(bucket_prob - bucket_label) * (mask.sum() / len(labels))
    return float(total)


def summarize_classification(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        return {"accuracy": math.nan, "log_loss": math.nan, "brier_score": math.nan, "ece": math.nan, "base_rate": math.nan, "rows": 0}
    return {
        "accuracy": binary_accuracy(labels, probabilities),
        "log_loss": binary_log_loss(labels, probabilities),
        "brier_score": brier_score(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
        "base_rate": float(labels.mean()),
        "rows": int(len(labels)),
    }


def simulate_backtest(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "predicted_yes_probability",
    market_column: str = "market_yes_probability",
    label_column: str = "label",
    edge_threshold: float = 0.05,
) -> dict[str, Any]:
    trades = []
    for row in frame.itertuples(index=False):
        edge = getattr(row, prediction_column) - getattr(row, market_column)
        label = int(getattr(row, label_column))
        market_probability = float(getattr(row, market_column))
        if edge >= edge_threshold:
            cost = market_probability
            payout = float(label)
            side = "yes"
        elif edge <= -edge_threshold:
            cost = 1.0 - market_probability
            payout = 1.0 - float(label)
            side = "no"
        else:
            continue
        pnl = payout - cost
        trades.append({"side": side, "edge": edge, "cost": cost, "pnl": pnl})

    if not trades:
        return {"trades": 0, "yes_trades": 0, "no_trades": 0, "roi": 0.0, "avg_pnl": 0.0, "win_rate": 0.0}

    trades_frame = pd.DataFrame.from_records(trades)
    total_cost = float(trades_frame["cost"].sum())
    total_pnl = float(trades_frame["pnl"].sum())
    return {
        "trades": int(len(trades_frame)),
        "yes_trades": int((trades_frame["side"] == "yes").sum()),
        "no_trades": int((trades_frame["side"] == "no").sum()),
        "roi": total_pnl / total_cost if total_cost else 0.0,
        "avg_pnl": float(trades_frame["pnl"].mean()),
        "win_rate": float((trades_frame["pnl"] > 0).mean()),
        "avg_abs_edge": float(trades_frame["edge"].abs().mean()),
    }
