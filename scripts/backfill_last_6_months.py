from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polymarket_predictor.api.client import ClobClient, GammaClient
from polymarket_predictor.datasets.backfill import build_horizon_dataset, extract_history_market_id
from polymarket_predictor.datasets.features import infer_resolution_label, parse_datetime
from polymarket_predictor.ml.pipeline import train_models

LOGGER = logging.getLogger("backfill_last_6_months")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill closed Polymarket markets from the last six months in smaller, resumable increments and train models."
    )
    parser.add_argument("--artifact-dir", default="artifacts/six_month_backfill")
    parser.add_argument("--days", type=int, default=183, help="How many days back to include. Default is roughly six months.")
    parser.add_argument("--page-size", type=int, default=100, help="Markets requested per Gamma page.")
    parser.add_argument("--max-pages", type=int, default=250, help="Hard cap on Gamma pages scanned before stopping.")
    parser.add_argument("--market-retries", type=int, default=5, help="Retries per Gamma page before giving up.")
    parser.add_argument("--price-retries", type=int, default=4, help="Retries per CLOB price-history request before giving up.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0, help="Base backoff between retries.")
    parser.add_argument("--page-delay-seconds", type=float, default=0.2, help="Small pause between successful page requests.")
    parser.add_argument("--price-delay-seconds", type=float, default=0.05, help="Small pause between successful price-history requests.")
    parser.add_argument("--gamma-timeout-seconds", type=int, default=30)
    parser.add_argument("--clob-timeout-seconds", type=int, default=30)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--interval", default="1d", help="Price-history interval for horizon construction.")
    parser.add_argument("--fidelity", type=int, default=60, help="Price-history fidelity in minutes when supported by the endpoint.")
    parser.add_argument("--max-price-markets", type=int, default=None, help="Optional cap on how many markets receive price-history backfill.")
    parser.add_argument("--horizon-hours", nargs="+", type=int, default=[24, 168, 720])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-category-samples", type=int, default=40)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["logistic", "boosted_trees"],
        choices=["logistic", "boosted_trees", "prior"],
        help="Model families to train after the dataset is built.",
    )
    return parser.parse_args()


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def market_sort_time(market: dict[str, Any]) -> datetime | None:
    for candidate in [market.get("updatedAt"), market.get("endDate"), market.get("createdAt")]:
        parsed = parse_datetime(candidate)
        if parsed is not None:
            return parsed
    return None


def market_key(market: dict[str, Any]) -> tuple[str, str]:
    return (
        str(market.get("id") or ""),
        str(market.get("updatedAt") or market.get("endDate") or ""),
    )


def price_history_key(payload: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(payload.get("history_market_id") or payload.get("condition_id") or ""),
        str(payload.get("interval") or ""),
        str(payload.get("fidelity") or ""),
    )


def skip_key(payload: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(payload.get("history_market_id") or ""),
        str(payload.get("interval") or ""),
        str(payload.get("fidelity") or ""),
    )


def request_with_retries(callback, *, retries: int, base_sleep_seconds: float, label: str):
    attempt = 0
    while True:
        attempt += 1
        try:
            return callback()
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                raise RuntimeError(f"{label} failed with non-retryable HTTP {status_code}: {exc}") from exc
            if attempt >= retries:
                raise RuntimeError(f"{label} failed after {attempt} attempts: {exc}") from exc
            sleep_seconds = base_sleep_seconds * attempt
            LOGGER.warning("%s failed on attempt %s/%s: %s. Retrying in %.1fs...", label, attempt, retries, exc, sleep_seconds)
            time.sleep(sleep_seconds)


def backfill_recent_markets(args: argparse.Namespace, *, cutoff: datetime, output_path: Path) -> list[dict[str, Any]]:
    client = GammaClient(timeout_seconds=args.gamma_timeout_seconds)
    existing_payloads = load_jsonl(output_path)
    existing_keys = {market_key(payload["market"]) for payload in existing_payloads}
    collected_payloads = list(existing_payloads)
    seen_new = 0
    recent_seen = 0

    LOGGER.info("Scanning closed markets updated since %s ...", cutoff.isoformat())
    for page_index in range(args.max_pages):
        offset = page_index * args.page_size

        def fetch_page() -> list[dict[str, Any]]:
            return client.list_markets(
                limit=args.page_size,
                offset=offset,
                closed=True,
                archived=False,
                order="updatedAt",
                ascending=False,
            )

        batch = request_with_retries(
            fetch_page,
            retries=args.market_retries,
            base_sleep_seconds=args.retry_backoff_seconds,
            label=f"Gamma page offset={offset}",
        )
        if not batch:
            LOGGER.info("No more closed markets returned.")
            break

        page_records: list[dict[str, Any]] = []
        newest_on_page: datetime | None = None
        oldest_on_page: datetime | None = None
        recent_count = 0
        for market in batch:
            sort_time = market_sort_time(market)
            if sort_time is not None:
                newest_on_page = sort_time if newest_on_page is None else max(newest_on_page, sort_time)
                oldest_on_page = sort_time if oldest_on_page is None else min(oldest_on_page, sort_time)
            if sort_time is not None and sort_time < cutoff:
                continue
            recent_count += 1
            recent_seen += 1
            key = market_key(market)
            if key in existing_keys:
                continue
            record = {"fetched_at": utc_now().isoformat(), "market": market}
            page_records.append(record)
            existing_keys.add(key)

        append_jsonl(output_path, page_records)
        collected_payloads.extend(page_records)
        seen_new += len(page_records)
        LOGGER.info(
            "Page %s: kept %s/%s recent markets, wrote %s new rows.",
            page_index + 1,
            recent_count,
            len(batch),
            len(page_records),
        )

        if oldest_on_page is not None and oldest_on_page < cutoff:
            LOGGER.info("Reached markets older than the six-month cutoff. Stopping market scan.")
            break
        if len(batch) < args.page_size:
            LOGGER.info("Received a short final page from Gamma. Stopping market scan.")
            break
        time.sleep(args.page_delay_seconds)

    LOGGER.info("Market backfill complete. Total new rows written: %s", seen_new)
    LOGGER.info("Recent closed markets scanned in range: %s", recent_seen)
    return collected_payloads


def history_window_for_market(market: dict[str, Any]) -> tuple[int | None, int | None]:
    start = parse_datetime(market.get("startDate") or market.get("createdAt"))
    end = parse_datetime(market.get("endDate") or market.get("updatedAt"))
    return (
        int(start.timestamp()) if start is not None else None,
        int(end.timestamp()) if end is not None else None,
    )


def fetch_price_history_with_fallbacks(
    client: ClobClient,
    *,
    history_market_id: str,
    start_ts: int | None,
    end_ts: int | None,
    interval: str,
    fidelity: int,
) -> dict[str, Any]:
    attempts = [
        {"start_ts": start_ts, "end_ts": end_ts, "interval": None, "fidelity": fidelity},
        {"start_ts": start_ts, "end_ts": end_ts, "interval": None, "fidelity": None},
        {"start_ts": None, "end_ts": None, "interval": interval, "fidelity": fidelity},
        {"start_ts": None, "end_ts": None, "interval": interval, "fidelity": None},
    ]
    last_error: Exception | None = None
    for index, params in enumerate(attempts, start=1):
        try:
            request_params: dict[str, Any] = {"market": history_market_id}
            if params["start_ts"] is not None:
                request_params["startTs"] = params["start_ts"]
            if params["end_ts"] is not None:
                request_params["endTs"] = params["end_ts"]
            if params["interval"] is not None:
                request_params["interval"] = params["interval"]
            if params["fidelity"] is not None:
                request_params["fidelity"] = params["fidelity"]

            prepared = client.session.prepare_request(
                requests.Request("GET", f"{client.base_url}/prices-history", params=request_params)
            )
            LOGGER.debug(
                "Price-history request attempt %s/%s market=%s url=%s params=%s",
                index,
                len(attempts),
                history_market_id,
                prepared.url,
                request_params,
            )
            response = client.session.send(prepared, timeout=client.timeout_seconds)
            LOGGER.debug(
                "Price-history response market=%s status=%s body=%s",
                history_market_id,
                response.status_code,
                response.text[:1000],
            )
            response.raise_for_status()
            payload = response.json()
            LOGGER.info(
                "Price-history request succeeded for market=%s on attempt %s/%s with %s history points.",
                history_market_id,
                index,
                len(attempts),
                len(payload.get("history", [])),
            )
            return payload
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            if response is not None:
                LOGGER.debug(
                    "Price-history HTTP error market=%s status=%s body=%s",
                    history_market_id,
                    response.status_code,
                    response.text[:1000],
                )
            if getattr(response, "status_code", None) != 400 or index == len(attempts):
                raise
            LOGGER.info(
                "Price-history fallback %s/%s for market=%s returned HTTP 400, trying a simpler parameter set...",
                index,
                len(attempts),
                history_market_id,
            )
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to fetch price history for market={history_market_id}")


def backfill_price_history_for_recent_markets(
    args: argparse.Namespace,
    *,
    recent_payloads: list[dict[str, Any]],
    cutoff: datetime,
    output_path: Path,
) -> None:
    client = ClobClient(timeout_seconds=args.clob_timeout_seconds)
    existing_payloads = load_jsonl(output_path)
    existing_keys = {price_history_key(payload) for payload in existing_payloads}
    skip_path = output_path.with_name("price_history_skips_6mo.jsonl")
    skipped_payloads = load_jsonl(skip_path)
    skipped_keys = {skip_key(payload) for payload in skipped_payloads}
    written = 0
    attempted = 0
    skipped = 0
    eligible = 0
    already_done = 0
    already_skipped = 0

    for payload in recent_payloads:
        market = payload["market"]
        resolved_at = market_sort_time(market)
        if resolved_at is None or resolved_at < cutoff:
            continue
        if infer_resolution_label(market) is None:
            continue
        if not bool(market.get("enableOrderBook")):
            continue
        history_market_id = extract_history_market_id(market)
        if not history_market_id:
            continue
        eligible += 1
        key = (history_market_id, args.interval, str(args.fidelity))
        if key in existing_keys:
            already_done += 1
            continue
        if key in skipped_keys:
            already_skipped += 1
            continue
        if args.max_price_markets is not None and attempted >= args.max_price_markets:
            break

        start_ts, end_ts = history_window_for_market(market)

        def fetch_history() -> dict[str, Any]:
            return fetch_price_history_with_fallbacks(
                client,
                history_market_id=history_market_id,
                start_ts=start_ts,
                end_ts=end_ts,
                interval=args.interval,
                fidelity=args.fidelity,
            )

        try:
            history_response = request_with_retries(
                fetch_history,
                retries=args.price_retries,
                base_sleep_seconds=args.retry_backoff_seconds,
                label=f"Price history market={history_market_id}",
            )
        except RuntimeError as exc:
            append_jsonl(
                skip_path,
                [
                    {
                        "market_id": str(market.get("id") or ""),
                        "history_market_id": history_market_id,
                        "condition_id": str(market.get("conditionId") or ""),
                        "interval": args.interval,
                        "fidelity": args.fidelity,
                        "reason": str(exc),
                    }
                ],
            )
            skipped_keys.add(key)
            skipped += 1
            LOGGER.warning("Skipping unsupported history market %s: %s", history_market_id, exc)
            continue

        append_jsonl(
            output_path,
            [
                {
                    "market_id": str(market.get("id") or ""),
                    "history_market_id": history_market_id,
                    "condition_id": str(market.get("conditionId") or ""),
                    "interval": args.interval,
                    "fidelity": args.fidelity,
                    "market": market,
                    "history": history_response.get("history", []),
                }
            ],
        )
        LOGGER.debug(
            "Wrote price-history payload for market=%s with %s history points to %s",
            history_market_id,
            len(history_response.get("history", [])),
            output_path,
        )
        existing_keys.add(key)
        attempted += 1
        written += 1
        if written % 25 == 0:
            LOGGER.info("Fetched price history for %s markets so far...", written)
        time.sleep(args.price_delay_seconds)

    LOGGER.info("Price-history backfill complete. New histories written: %s. Unsupported/skipped: %s", written, skipped)
    LOGGER.info(
        "Price-history coverage summary: eligible=%s, already_fetched=%s, already_marked_unsupported=%s, new_fetched=%s, new_unsupported=%s",
        eligible,
        already_done,
        already_skipped,
        written,
        skipped,
    )
    return {
        "eligible_orderbook_markets": eligible,
        "already_fetched_histories": already_done,
        "already_marked_unsupported": already_skipped,
        "new_histories_written": written,
        "new_unsupported_skips": skipped,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cutoff = utc_now() - timedelta(days=args.days)

    market_snapshot_path = artifact_dir / "raw_closed_history_6mo.jsonl"
    price_history_path = artifact_dir / "price_history_6mo.jsonl"
    skip_path = artifact_dir / "price_history_skips_6mo.jsonl"
    horizon_dataset_path = artifact_dir / "horizon_training_data_6mo.csv"
    summary_path = artifact_dir / "run_summary.json"

    recent_payloads = backfill_recent_markets(args, cutoff=cutoff, output_path=market_snapshot_path)
    recent_closed_market_count = 0
    for payload in recent_payloads:
        sort_time = market_sort_time(payload["market"])
        if sort_time is not None and sort_time >= cutoff:
            recent_closed_market_count += 1

    price_history_summary = backfill_price_history_for_recent_markets(
        args,
        recent_payloads=recent_payloads,
        cutoff=cutoff,
        output_path=price_history_path,
    )

    horizon_result = build_horizon_dataset(
        price_history_path=price_history_path,
        output_path=horizon_dataset_path,
        horizon_hours=args.horizon_hours,
    )
    LOGGER.info("Horizon dataset written to %s with %s rows.", horizon_result.output_path, horizon_result.records_written)

    training_summaries: dict[str, Any] = {}
    for model_type in args.model_types:
        run_dir = artifact_dir / f"{model_type}_6mo"
        result = train_models(
            artifact_dir=run_dir,
            dataset_path=horizon_dataset_path,
            validation_fraction=args.validation_fraction,
            min_category_samples=args.min_category_samples,
            model_type=model_type,
            edge_threshold=args.edge_threshold,
        )
        validation = result.metrics.get("global_validation", {})
        training_summaries[model_type] = {
            "artifact_dir": str(run_dir),
            "validation_log_loss": validation.get("log_loss"),
            "validation_brier_score": validation.get("brier_score"),
            "validation_ece": validation.get("ece"),
            "training_rows": result.metrics.get("training_rows"),
            "validation_rows": result.metrics.get("validation_rows"),
        }
        LOGGER.info(
            "Trained %s: log_loss=%s, brier=%s, ece=%s",
            model_type,
            validation.get("log_loss"),
            validation.get("brier_score"),
            validation.get("ece"),
        )

    summary = {
        "cutoff_utc": cutoff.isoformat(),
        "market_snapshot_path": str(market_snapshot_path),
        "price_history_path": str(price_history_path),
        "price_history_skip_path": str(skip_path),
        "horizon_dataset_path": str(horizon_dataset_path),
        "counts": {
            "recent_closed_markets_in_range": recent_closed_market_count,
            **price_history_summary,
            "horizon_rows_written": horizon_result.records_written,
        },
        "trained_models": training_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Run summary counts:\n%s", json.dumps(summary["counts"], indent=2))
    LOGGER.info("Run summary written to %s", summary_path)


if __name__ == "__main__":
    main()
