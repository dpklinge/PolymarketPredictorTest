from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .taxonomy import CategoryTaxonomy, load_taxonomy


TEXT_DIMENSIONS = 32
TEMPORAL_DIMENSIONS = 8
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


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


def normalize_category(market: dict[str, Any], taxonomy: CategoryTaxonomy | None = None) -> str:
    taxonomy = taxonomy or load_taxonomy()
    direct = taxonomy.normalize(market.get("category") or "")
    if direct:
        return direct

    events = market.get("events") or []
    for event in events:
        category = taxonomy.normalize(event.get("category") or "")
        if category:
            return category

    tags = market.get("tags") or []
    selected = _select_category_from_tags(tags, taxonomy)
    if selected:
        return selected

    for tag in tags:
        label = taxonomy.normalize(tag.get("label") or tag.get("slug") or "")
        if label:
            return label

    for event in events:
        selected = _select_category_from_tags(event.get("tags") or [], taxonomy)
        if selected:
            return selected
    return "unknown"


def _select_category_from_tags(tags: list[dict[str, Any]], taxonomy: CategoryTaxonomy) -> str | None:
    normalized: list[str] = []
    for tag in tags:
        label = taxonomy.normalize(tag.get("slug") or tag.get("label") or "")
        if label:
            normalized.append(label)

    for label in normalized:
        if label in taxonomy.preferred_categories:
            return label

    for label in normalized:
        if label not in taxonomy.generic_tags:
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


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator


def _event_aggregates(market: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    events = market.get("events") or []
    if not events:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    event_volumes = [parse_float(event.get("volume")) for event in events]
    event_liquidities = [parse_float(event.get("liquidity")) for event in events]
    event_open_interest = [parse_float(event.get("openInterest")) for event in events]
    event_comments = [parse_float(event.get("commentCount")) for event in events]
    event_competitiveness = [parse_float(event.get("competitive")) for event in events]
    event_tag_counts = [float(len(event.get("tags") or [])) for event in events]
    return (
        math.log1p(sum(event_volumes)),
        math.log1p(sum(event_liquidities)),
        math.log1p(sum(event_open_interest)),
        math.log1p(sum(event_comments)),
        float(np.mean(event_competitiveness)),
        float(np.mean(event_tag_counts)),
    )


def _text_summary_features(question: str, description: str) -> np.ndarray:
    combined = f"{question} {description}".strip()
    tokens = TOKEN_PATTERN.findall(combined)
    unique_tokens = len(set(token.lower() for token in tokens))
    question_tokens = TOKEN_PATTERN.findall(question)
    description_tokens = TOKEN_PATTERN.findall(description)
    uppercase_chars = sum(1 for char in question if char.isupper())

    return np.array(
        [
            float(len(tokens)),
            float(unique_tokens),
            _safe_ratio(unique_tokens, max(len(tokens), 1)),
            float(len(question_tokens)),
            float(len(description_tokens)),
            float(question.count("?")),
            float(question.lower().startswith("will ")),
            _safe_ratio(uppercase_chars, max(len(question), 1)),
        ],
        dtype=float,
    )


@dataclass
class FeatureRow:
    market_id: str
    slug: str
    question: str
    category: str
    market_yes_probability: float
    model_features: np.ndarray
    label: int | None


def default_temporal_features() -> np.ndarray:
    return np.zeros(TEMPORAL_DIMENSIONS, dtype=float)


def build_feature_row(
    market: dict[str, Any],
    *,
    now: datetime | None = None,
    taxonomy: CategoryTaxonomy | None = None,
    temporal_features: np.ndarray | None = None,
) -> FeatureRow | None:
    now = now or utc_now()
    taxonomy = taxonomy or load_taxonomy()
    yes_probability = extract_yes_probability(market)
    if yes_probability is None:
        return None

    question = market.get("question") or ""
    description = market.get("description") or ""
    text_features = hash_text(f"{question}\n{description}")
    text_summary_features = _text_summary_features(question, description)
    events = market.get("events") or []
    tags = market.get("tags") or []

    start_date = parse_datetime(market.get("startDate") or market.get("startDateIso"))
    end_date = parse_datetime(market.get("endDate") or market.get("endDateIso"))
    created_at = parse_datetime(market.get("createdAt"))

    reference_start = start_date or created_at or now
    reference_end = end_date or now

    age_hours = max((now - reference_start).total_seconds() / 3600.0, 0.0)
    time_to_close_hours = (reference_end - now).total_seconds() / 3600.0
    duration_hours = max((reference_end - reference_start).total_seconds() / 3600.0, 0.0)
    elapsed_ratio = 0.0 if duration_hours <= 0 else min(max(age_hours / max(duration_hours, 1e-6), 0.0), 2.0)
    best_bid = parse_float(market.get("bestBid"))
    best_ask = parse_float(market.get("bestAsk"))
    spread = parse_float(market.get("spread"))
    last_trade_price = parse_float(market.get("lastTradePrice"), default=yes_probability)
    mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else yes_probability
    event_volume, event_liquidity, event_open_interest, event_comments, event_competitiveness, event_avg_tag_count = (
        _event_aggregates(market)
    )

    numeric_features = np.array(
        [
            yes_probability,
            abs(yes_probability - 0.5),
            safe_logit(yes_probability),
            math.log1p(parse_float(market.get("volumeNum") or market.get("volume"))),
            math.log1p(parse_float(market.get("liquidityNum") or market.get("liquidity"))),
            math.log1p(parse_float(market.get("volume24hr"))),
            math.log1p(parse_float(market.get("volume1wk"))),
            math.log1p(parse_float(market.get("volume1mo"))),
            math.log1p(max(age_hours, 0.0)),
            math.log1p(max(time_to_close_hours, 0.0)),
            math.log1p(max(duration_hours, 0.0)),
            elapsed_ratio,
            spread,
            best_bid,
            best_ask,
            last_trade_price,
            mid_price,
            abs(last_trade_price - yes_probability),
            _safe_ratio(spread, max(mid_price, 1e-6)),
            _safe_ratio(best_ask - yes_probability, max(best_ask, 1e-6)),
            _safe_ratio(yes_probability - best_bid, max(yes_probability, 1e-6)),
            parse_float(market.get("oneHourPriceChange")),
            parse_float(market.get("oneDayPriceChange")),
            parse_float(market.get("oneWeekPriceChange")),
            parse_float(market.get("oneMonthPriceChange")),
            parse_float(market.get("competitive")),
            parse_float(market.get("groupItemThreshold")),
            event_volume,
            event_liquidity,
            event_open_interest,
            event_comments,
            event_competitiveness,
            event_avg_tag_count,
            float(bool(market.get("active"))),
            float(bool(market.get("closed"))),
            float(bool(market.get("featured"))),
            float(bool(market.get("enableOrderBook"))),
            float(bool(market.get("automaticallyResolved"))),
            float(bool(market.get("acceptingOrders"))),
            float(bool(market.get("new"))),
            float(bool(market.get("negRisk"))),
            float(bool(market.get("restricted"))),
            float(bool(market.get("ready"))),
            float(bool(market.get("funded"))),
            float(len(events)),
            float(len(tags)),
            float(len(question)),
            float(len(description)),
        ],
        dtype=float,
    )
    temporal_features = temporal_features if temporal_features is not None else default_temporal_features()

    return FeatureRow(
        market_id=str(market.get("id") or ""),
        slug=str(market.get("slug") or ""),
        question=question,
        category=normalize_category(market, taxonomy),
        market_yes_probability=yes_probability,
        model_features=np.concatenate([numeric_features, text_summary_features, temporal_features, text_features]),
        label=infer_resolution_label(market),
    )
