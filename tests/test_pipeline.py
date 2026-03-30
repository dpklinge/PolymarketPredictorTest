from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from polymarket_predictor.features import build_feature_row
from polymarket_predictor.model import LogisticModel


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
