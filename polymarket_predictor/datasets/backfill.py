from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..api.client import ClobClient, GammaClient
from .data import utc_now_iso
from .features import build_feature_row, infer_resolution_label, parse_datetime, parse_jsonish_list
from .taxonomy import load_taxonomy


DEFAULT_HORIZONS_HOURS = [24, 24 * 7, 24 * 30]


@dataclass
class BackfillResult:
    output_path: Path
    records_written: int


def extract_history_market_id(market: dict[str, Any]) -> str:
    for field in ["clobTokenIds", "tokenIds"]:
        values = parse_jsonish_list(market.get(field))
        if values:
            return str(values[0])

    tokens = market.get("tokens")
    if isinstance(tokens, list) and tokens:
        first = tokens[0]
        if isinstance(first, dict):
            for key in ["tokenId", "id", "asset_id"]:
                value = first.get(key)
                if value:
                    return str(value)
        elif first:
            return str(first)

    for key in ["yesTokenId", "clobTokenId", "tokenId", "conditionId"]:
        value = market.get(key)
        if value:
            return str(value)
    return ""


def backfill_closed_markets(
    *,
    output_path: str | Path,
    max_pages: int,
    page_size: int,
    order: str = "updatedAt",
    ascending: bool = False,
    append: bool = True,
) -> BackfillResult:
    client = GammaClient()
    markets = client.fetch_market_pages(
        closed=True,
        max_pages=max_pages,
        page_size=page_size,
        order=order,
        ascending=ascending,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = _existing_market_backfill_keys(destination) if append and destination.exists() else set()
    mode = "a" if append and destination.exists() else "w"
    fetched_at = utc_now_iso()
    written = 0
    with destination.open(mode, encoding="utf-8") as handle:
        for market in markets:
            key = _market_backfill_key(market, fetched_at)
            if key in existing_keys:
                continue
            handle.write(json.dumps({"fetched_at": fetched_at, "market": market}) + "\n")
            existing_keys.add(key)
            written += 1
    return BackfillResult(output_path=destination, records_written=written)


def backfill_price_history(
    *,
    market_snapshot_path: str | Path,
    output_path: str | Path,
    interval: str = "1d",
    fidelity: int = 60,
    max_markets: int | None = None,
    append: bool = False,
) -> BackfillResult:
    client = ClobClient()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = _existing_price_history_keys(destination) if append and destination.exists() else set()
    mode = "a" if append and destination.exists() else "w"

    payloads = _load_jsonl(market_snapshot_path)
    written = 0
    with destination.open(mode, encoding="utf-8") as handle:
        for payload in payloads:
            market = payload["market"]
            if infer_resolution_label(market) is None:
                continue
            history_market_id = extract_history_market_id(market)
            if not history_market_id:
                continue
            key = (history_market_id, interval, str(fidelity))
            if key in existing_keys:
                continue
            start_ts, end_ts = _history_window_for_market(market)
            history = client.get_prices_history(
                market=history_market_id,
                start_ts=start_ts,
                end_ts=end_ts,
                interval=None,
                fidelity=fidelity,
            )
            handle.write(
                json.dumps(
                    {
                        "market_id": str(market.get("id") or ""),
                        "history_market_id": history_market_id,
                        "condition_id": str(market.get("conditionId") or ""),
                        "interval": interval,
                        "fidelity": fidelity,
                        "market": market,
                        "history": history.get("history", []),
                    }
                )
                + "\n"
            )
            written += 1
            existing_keys.add(key)
            if max_markets is not None and written >= max_markets:
                break
    return BackfillResult(output_path=destination, records_written=written)


def build_horizon_dataset(
    *,
    price_history_path: str | Path,
    output_path: str | Path,
    horizon_hours: list[int] | None = None,
) -> BackfillResult:
    taxonomy = load_taxonomy()
    horizons = horizon_hours or list(DEFAULT_HORIZONS_HOURS)
    records: list[dict[str, Any]] = []

    for payload in _load_jsonl(price_history_path):
        market = payload["market"]
        label = infer_resolution_label(market)
        if label is None:
            continue
        resolved_at = _market_resolved_at(market)
        if resolved_at is None:
            continue
        history = payload.get("history") or []
        if not history:
            continue

        for horizon in horizons:
            anchor_time = resolved_at - timedelta(hours=horizon)
            price_point = _latest_price_before(history, anchor_time)
            if price_point is None:
                continue
            synthetic_market = _market_at_horizon(market, price_point["p"])
            row = build_feature_row(synthetic_market, now=anchor_time, taxonomy=taxonomy)
            if row is None:
                continue
            records.append(
                {
                    "market_id": row.market_id,
                    "slug": row.slug,
                    "question": row.question,
                    "category": row.category,
                    "market_yes_probability": row.market_yes_probability,
                    "label": label,
                    "horizon_hours": horizon,
                    "anchor_time": anchor_time.isoformat(),
                    "resolved_at": resolved_at.isoformat(),
                    "features": json.dumps(row.model_features.tolist()),
                }
            )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("No horizon rows could be built from the supplied price history.")
    frame = frame.drop_duplicates(subset=["market_id", "horizon_hours", "anchor_time"]).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return BackfillResult(output_path=destination, records_written=len(frame))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _market_backfill_key(market: dict[str, Any], fetched_at: str | None) -> tuple[str, str, str]:
    return (
        str(market.get("id") or ""),
        fetched_at or "",
        str(market.get("updatedAt") or market.get("endDate") or ""),
    )


def _existing_market_backfill_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for payload in _load_jsonl(path):
        keys.add(_market_backfill_key(payload["market"], payload.get("fetched_at")))
    return keys


def _existing_price_history_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for payload in _load_jsonl(path):
        keys.add(
            (
                str(payload.get("history_market_id") or payload.get("condition_id") or ""),
                str(payload.get("interval") or ""),
                str(payload.get("fidelity") or ""),
            )
        )
    return keys


def _history_window_for_market(market: dict[str, Any]) -> tuple[int | None, int | None]:
    start = parse_datetime(market.get("startDate") or market.get("createdAt"))
    end = parse_datetime(market.get("endDate") or market.get("updatedAt"))
    start_ts = int(start.timestamp()) if start else None
    end_ts = int(end.timestamp()) if end else None
    return start_ts, end_ts


def _market_resolved_at(market: dict[str, Any]) -> datetime | None:
    for candidate in [market.get("endDate"), market.get("updatedAt")]:
        parsed = parse_datetime(candidate)
        if parsed is not None:
            return parsed
    return None


def _latest_price_before(history: list[dict[str, Any]], anchor_time: datetime) -> dict[str, Any] | None:
    anchor_ts = anchor_time.timestamp()
    candidates = [point for point in history if float(point.get("t", 0)) <= anchor_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda point: float(point.get("t", 0)))


def _market_at_horizon(market: dict[str, Any], yes_price: float) -> dict[str, Any]:
    synthetic = deepcopy(market)
    clipped_price = min(max(float(yes_price), 0.0), 1.0)
    synthetic["outcomePrices"] = json.dumps([clipped_price, 1.0 - clipped_price])
    synthetic["lastTradePrice"] = clipped_price
    synthetic["bestBid"] = synthetic.get("bestBid") or clipped_price
    synthetic["bestAsk"] = synthetic.get("bestAsk") or clipped_price
    return synthetic
