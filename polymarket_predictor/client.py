from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


@dataclass
class GammaClient:
    timeout_seconds: int = 30
    base_url: str = GAMMA_BASE_URL

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self._event_cache: dict[str, dict[str, Any]] = {}
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "polymarket-predictor/1.0",
            }
        )

    def list_markets(
        self,
        *,
        limit: int = 200,
        offset: int = 0,
        closed: bool | None = None,
        archived: bool | None = False,
        order: str | None = None,
        ascending: bool | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if closed is not None:
            params["closed"] = str(closed).lower()
        if archived is not None:
            params["archived"] = str(archived).lower()
        if order:
            params["order"] = order
        if ascending is not None:
            params["ascending"] = str(ascending).lower()
        return self._get_json("/markets", params=params)

    def fetch_market_pages(
        self,
        *,
        closed: bool,
        max_pages: int,
        page_size: int,
        order: str | None = None,
        ascending: bool | None = None,
        enrich_event_tags: bool = True,
    ) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        for page in range(max_pages):
            batch = self.list_markets(
                limit=page_size,
                offset=page * page_size,
                closed=closed,
                archived=False,
                order=order,
                ascending=ascending,
            )
            if not batch:
                break
            if enrich_event_tags:
                batch = [self.enrich_market_with_event_tags(market) for market in batch]
            markets.extend(batch)
            if len(batch) < page_size:
                break
        return markets

    def get_event(self, event_id: str) -> dict[str, Any]:
        if event_id not in self._event_cache:
            self._event_cache[event_id] = self._get_json(f"/events/{event_id}", params={})
        return self._event_cache[event_id]

    def enrich_market_with_event_tags(self, market: dict[str, Any]) -> dict[str, Any]:
        if market.get("category") or market.get("tags"):
            return market

        events = market.get("events") or []
        if not events:
            return market

        enriched_market = dict(market)
        enriched_events: list[dict[str, Any]] = []
        aggregate_tags: list[dict[str, Any]] = []

        for event in events:
            event_id = event.get("id")
            if not event_id:
                enriched_events.append(event)
                continue
            event_details = self.get_event(str(event_id))
            merged_event = dict(event)
            merged_event["tags"] = event_details.get("tags") or []
            merged_event["title"] = event_details.get("title") or event.get("title")
            aggregate_tags.extend(merged_event["tags"])
            enriched_events.append(merged_event)

        enriched_market["events"] = enriched_events
        if aggregate_tags:
            enriched_market["tags"] = aggregate_tags
        return enriched_market

    def _get_json(self, path: str, *, params: dict[str, Any]) -> Any:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
