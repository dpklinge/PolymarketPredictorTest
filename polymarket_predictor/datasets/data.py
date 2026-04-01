from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..api.client import GammaClient
from .features import TEMPORAL_DIMENSIONS, build_feature_row, parse_datetime, parse_float
from .taxonomy import CategoryTaxonomy, load_taxonomy


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_snapshots(
    *,
    output_path: str | Path,
    closed: bool,
    max_pages: int,
    page_size: int,
    order: str | None,
    ascending: bool | None,
    append: bool = False,
) -> Path:
    client = GammaClient()
    markets = client.fetch_market_pages(
        closed=closed,
        max_pages=max_pages,
        page_size=page_size,
        order=order,
        ascending=ascending,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetched_at = utc_now_iso()
    existing_keys = _existing_snapshot_keys(destination) if append and destination.exists() else set()
    mode = "a" if append and destination.exists() else "w"
    with destination.open(mode, encoding="utf-8") as handle:
        for market in markets:
            key = _snapshot_key(market, fetched_at)
            if key in existing_keys:
                continue
            handle.write(json.dumps({"fetched_at": fetched_at, "market": market}) + "\n")
            existing_keys.add(key)
    return destination


def load_snapshot_file(path: str | Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payloads.append(json.loads(line))
    return payloads


def _snapshot_key(market: dict[str, Any], fetched_at: str | None) -> tuple[str, str, str]:
    return (
        str(market.get("id") or ""),
        fetched_at or "",
        str(market.get("updatedAt") or market.get("endDate") or ""),
    )


def _existing_snapshot_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for payload in load_snapshot_file(path):
        keys.add(_snapshot_key(payload["market"], payload.get("fetched_at")))
    return keys


def _resolved_at(market: dict[str, Any], fetched_at: str | None) -> datetime | None:
    for candidate in [market.get("endDate"), market.get("updatedAt"), fetched_at]:
        parsed = parse_datetime(candidate)
        if parsed is not None:
            return parsed
    return None


def _build_temporal_features(
    market: dict[str, Any],
    *,
    fetched_at: str | None,
    previous_market: dict[str, Any] | None,
    previous_fetched_at: str | None,
) -> np.ndarray:
    current_price = parse_float(parse_json_price(market), default=0.0)
    current_volume = parse_float(market.get("volumeNum") or market.get("volume"))
    current_liquidity = parse_float(market.get("liquidityNum") or market.get("liquidity"))

    if previous_market is None:
        return np.zeros(TEMPORAL_DIMENSIONS, dtype=float)

    previous_price = parse_float(parse_json_price(previous_market), default=current_price)
    previous_volume = parse_float(previous_market.get("volumeNum") or previous_market.get("volume"))
    previous_liquidity = parse_float(previous_market.get("liquidityNum") or previous_market.get("liquidity"))

    current_time = parse_datetime(fetched_at)
    previous_time = parse_datetime(previous_fetched_at)
    delta_hours = 0.0
    if current_time and previous_time:
        delta_hours = max((current_time - previous_time).total_seconds() / 3600.0, 0.0)

    price_delta = current_price - previous_price
    volume_delta = current_volume - previous_volume
    liquidity_delta = current_liquidity - previous_liquidity
    price_velocity = price_delta / delta_hours if delta_hours > 0.0 else 0.0

    def _signed_log1p(x: float) -> float:
        return math.copysign(math.log1p(abs(x)), x)

    return np.array(
        [
            previous_price,
            price_delta,
            abs(price_delta),
            price_velocity,
            _signed_log1p(volume_delta),
            _signed_log1p(liquidity_delta),
            math.log1p(delta_hours),
            float(previous_market is not None),
        ],
        dtype=float,
    )


def parse_json_price(market: dict[str, Any]) -> Any:
    outcome_prices = market.get("outcomePrices")
    if isinstance(outcome_prices, list) and outcome_prices:
        return outcome_prices[0]
    if isinstance(outcome_prices, str):
        try:
            parsed = json.loads(outcome_prices)
            if isinstance(parsed, list) and parsed:
                return parsed[0]
        except json.JSONDecodeError:
            return None
    return None


def prepare_dataset(
    *,
    snapshot_paths: list[str | Path],
    output_path: str | Path,
    taxonomy: CategoryTaxonomy | None = None,
) -> Path:
    taxonomy = taxonomy or load_taxonomy()
    records: list[dict[str, Any]] = []
    payloads_with_time: list[dict[str, Any]] = []

    for snapshot_path in snapshot_paths:
        payloads_with_time.extend(load_snapshot_file(snapshot_path))

    payloads_with_time.sort(
        key=lambda payload: (
            str(payload["market"].get("id") or ""),
            payload.get("fetched_at") or "",
        )
    )

    last_seen_by_market: dict[str, tuple[dict[str, Any], str | None]] = {}
    for payload in payloads_with_time:
        market = payload["market"]
        fetched_at = payload.get("fetched_at")
        market_id = str(market.get("id") or "")
        previous_market, previous_fetched_at = last_seen_by_market.get(market_id, (None, None))
        temporal_features = _build_temporal_features(
            market,
            fetched_at=fetched_at,
            previous_market=previous_market,
            previous_fetched_at=previous_fetched_at,
        )
        row = build_feature_row(market, taxonomy=taxonomy, temporal_features=temporal_features)
        if row is None:
            continue
        resolved_at = _resolved_at(market, fetched_at)
        records.append(
            {
                "market_id": row.market_id,
                "slug": row.slug,
                "question": row.question,
                "category": row.category,
                "market_yes_probability": row.market_yes_probability,
                "label": row.label,
                "fetched_at": fetched_at or "",
                "resolved_at": resolved_at.isoformat() if resolved_at else "",
                "features": json.dumps(row.model_features.tolist()),
            }
        )
        last_seen_by_market[market_id] = (market, fetched_at)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("No usable markets were found while preparing the dataset.")

    frame = frame.drop_duplicates(subset=["market_id", "fetched_at", "market_yes_probability"]).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def build_market_history_index(snapshot_paths: list[str | Path]) -> dict[str, tuple[dict[str, Any], str | None]]:
    payloads_with_time: list[dict[str, Any]] = []
    for snapshot_path in snapshot_paths:
        payloads_with_time.extend(load_snapshot_file(snapshot_path))

    payloads_with_time.sort(
        key=lambda payload: (
            str(payload["market"].get("id") or ""),
            payload.get("fetched_at") or "",
        )
    )

    history_index: dict[str, tuple[dict[str, Any], str | None]] = {}
    for payload in payloads_with_time:
        market = payload["market"]
        market_id = str(market.get("id") or "")
        history_index[market_id] = (market, payload.get("fetched_at"))
    return history_index


def temporal_features_from_history(
    market: dict[str, Any],
    *,
    fetched_at: str | None,
    history_index: dict[str, tuple[dict[str, Any], str | None]] | None,
) -> np.ndarray:
    if not history_index:
        return np.zeros(TEMPORAL_DIMENSIONS, dtype=float)
    market_id = str(market.get("id") or "")
    previous_market, previous_fetched_at = history_index.get(market_id, (None, None))
    return _build_temporal_features(
        market,
        fetched_at=fetched_at,
        previous_market=previous_market,
        previous_fetched_at=previous_fetched_at,
    )
