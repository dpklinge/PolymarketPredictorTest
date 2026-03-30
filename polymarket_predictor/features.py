from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


TEXT_DIMENSIONS = 32
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
GENERIC_TAGS = {"other", "featured", "breaking", "news"}
PREFERRED_CATEGORIES = {
    "politics",
    "crypto",
    "sports",
    "business",
    "tech",
    "technology",
    "science",
    "culture",
    "pop-culture",
    "world",
    "finance",
    "economics",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_jsonish_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def safe_logit(probability: float) -> float:
    clipped = min(max(probability, 1e-4), 1.0 - 1e-4)
    return math.log(clipped / (1.0 - clipped))


def normalize_category(market: dict[str, Any]) -> str:
    direct = (market.get("category") or "").strip().lower()
    if direct:
        return direct

    events = market.get("events") or []
    for event in events:
        category = (event.get("category") or "").strip().lower()
        if category:
            return category

    tags = market.get("tags") or []
    selected = _select_category_from_tags(tags)
    if selected:
        return selected

    for tag in tags:
        label = (tag.get("label") or tag.get("slug") or "").strip().lower()
        if label:
            return label

    for event in events:
        selected = _select_category_from_tags(event.get("tags") or [])
        if selected:
            return selected
    return "unknown"


def _select_category_from_tags(tags: list[dict[str, Any]]) -> str | None:
    normalized: list[str] = []
    for tag in tags:
        label = (tag.get("slug") or tag.get("label") or "").strip().lower()
        if label:
            normalized.append(label)

    for label in normalized:
        if label in PREFERRED_CATEGORIES:
            return label

    for label in normalized:
        if label not in GENERIC_TAGS:
            return label
    return None


def extract_yes_probability(market: dict[str, Any]) -> float | None:
    prices = [parse_float(value, default=float("nan")) for value in parse_jsonish_list(market.get("outcomePrices"))]
    if len(prices) < 2 or any(math.isnan(value) for value in prices[:2]):
        return None
    return min(max(prices[0], 0.0), 1.0)


def infer_resolution_label(market: dict[str, Any]) -> int | None:
    prices = [parse_float(value, default=float("nan")) for value in parse_jsonish_list(market.get("outcomePrices"))]
    if len(prices) != 2 or any(math.isnan(value) for value in prices):
        return None
    if prices[0] >= 0.999 and prices[1] <= 0.001:
        return 1
    if prices[1] >= 0.999 and prices[0] <= 0.001:
        return 0
    return None


def hash_text(text: str, dimensions: int = TEXT_DIMENSIONS) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=float)
    if not text:
        return vector
    for token in TOKEN_PATTERN.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


@dataclass
class FeatureRow:
    market_id: str
    slug: str
    question: str
    category: str
    market_yes_probability: float
    model_features: np.ndarray
    label: int | None


def build_feature_row(market: dict[str, Any], *, now: datetime | None = None) -> FeatureRow | None:
    now = now or utc_now()
    yes_probability = extract_yes_probability(market)
    if yes_probability is None:
        return None

    question = market.get("question") or ""
    description = market.get("description") or ""
    text_features = hash_text(f"{question}\n{description}")

    start_date = parse_datetime(market.get("startDate") or market.get("startDateIso"))
    end_date = parse_datetime(market.get("endDate") or market.get("endDateIso"))
    created_at = parse_datetime(market.get("createdAt"))

    reference_start = start_date or created_at or now
    reference_end = end_date or now

    age_hours = max((now - reference_start).total_seconds() / 3600.0, 0.0)
    time_to_close_hours = (reference_end - now).total_seconds() / 3600.0
    duration_hours = max((reference_end - reference_start).total_seconds() / 3600.0, 0.0)
    elapsed_ratio = 0.0 if duration_hours <= 0 else min(max(age_hours / max(duration_hours, 1e-6), 0.0), 2.0)

    numeric_features = np.array(
        [
            yes_probability,
            abs(yes_probability - 0.5),
            safe_logit(yes_probability),
            math.log1p(parse_float(market.get("volumeNum") or market.get("volume"))),
            math.log1p(parse_float(market.get("liquidityNum") or market.get("liquidity"))),
            math.log1p(max(age_hours, 0.0)),
            math.log1p(max(time_to_close_hours, 0.0)),
            math.log1p(max(duration_hours, 0.0)),
            elapsed_ratio,
            parse_float(market.get("spread")),
            parse_float(market.get("bestBid")),
            parse_float(market.get("bestAsk")),
            parse_float(market.get("lastTradePrice")),
            parse_float(market.get("oneHourPriceChange")),
            parse_float(market.get("oneDayPriceChange")),
            parse_float(market.get("oneWeekPriceChange")),
            parse_float(market.get("oneMonthPriceChange")),
            float(bool(market.get("active"))),
            float(bool(market.get("closed"))),
            float(bool(market.get("featured"))),
            float(bool(market.get("enableOrderBook"))),
            float(bool(market.get("automaticallyResolved"))),
            float(bool(market.get("acceptingOrders"))),
            float(bool(market.get("new"))),
            float(len(question)),
            float(len(description)),
        ],
        dtype=float,
    )

    return FeatureRow(
        market_id=str(market.get("id") or ""),
        slug=str(market.get("slug") or ""),
        question=question,
        category=normalize_category(market),
        market_yes_probability=yes_probability,
        model_features=np.concatenate([numeric_features, text_features]),
        label=infer_resolution_label(market),
    )
