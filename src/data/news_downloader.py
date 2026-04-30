"""Download historical news from Alpha Vantage NEWS_SENTIMENT endpoint.

Pagination strategy
-------------------
The endpoint returns at most 1000 articles per request.  We sort by EARLIEST
and advance ``time_from`` to (last_item.time_published + 1 min) after each
full page, repeating until a page returns fewer than 1000 items.

Resume strategy
---------------
After every successful request the accumulated list is written to
``data/raw/news/{ticker}_raw.json``.  On restart the file is loaded, the
maximum ``time_published`` is recovered, and fetching continues from there —
no page is re-requested.

Rate limit
----------
75 requests / minute  →  sleep ≥ 0.80 s between requests.
We use 0.85 s to leave a small margin.
"""
# Defers type-hint evaluation so ``str | None`` works on Python 3.9.
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

# ── Paths & constants ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
NEWS_DIR = ROOT / "data" / "raw" / "news"

BASE_URL = "https://www.alphavantage.co/query"
LIMIT = 1000
SLEEP_S = 0.85  # 75 req/min ≈ 0.80 s/req; 0.85 s gives ~70 req/min

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Timestamp helpers ──────────────────────────────────────────────────────────

def _to_av_ts(date_str: str, end_of_day: bool = False) -> str:
    """Convert ``YYYY-MM-DD`` to AV parameter format ``YYYYMMDDTHHMM``."""
    d = date_str.replace("-", "")
    return f"{d}T2359" if end_of_day else f"{d}T0000"


def _parse_published(ts: str) -> datetime:
    """Parse a ``time_published`` field from a feed item.

    AV returns ``YYYYMMDDTHHMMSS`` (with seconds) in feed items,
    but we also handle the parameter format ``YYYYMMDDTHHMM`` defensively.
    """
    ts = ts.strip()
    fmt = "%Y%m%dT%H%M%S" if len(ts) == 15 else "%Y%m%dT%H%M"
    return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)


def _advance_ts(published: str) -> str:
    """Return a ``YYYYMMDDTHHMM`` string 1 minute after *published*."""
    dt = _parse_published(published) + timedelta(minutes=1)
    return dt.strftime("%Y%m%dT%H%M")


# ── Persistence helpers ────────────────────────────────────────────────────────

def _load_existing(path: Path) -> tuple[list[dict], set[str], str | None]:
    """Load an existing progress file.

    Returns:
        items     – the full list already saved
        seen_urls – set of URLs used for deduplication at page boundaries
        resume_ts – AV-formatted timestamp to resume from (or None if fresh)
    """
    if not path.exists():
        return [], set(), None

    items: list[dict] = json.loads(path.read_text())
    seen_urls = {it.get("url", "") for it in items}

    published = [it["time_published"] for it in items if "time_published" in it]
    if not published:
        return items, seen_urls, None

    latest = max(published)
    resume_ts = _advance_ts(latest)
    return items, seen_urls, resume_ts


def _save(items: list[dict], path: Path) -> None:
    path.write_text(json.dumps(items, indent=2, default=str))


# ── Core fetcher ───────────────────────────────────────────────────────────────

def fetch_ticker_news(ticker: str, api_key: str) -> int:
    """Download all available news for *ticker* and save to disk.

    Returns the total number of items in the output file.
    """
    out = NEWS_DIR / f"{ticker}_raw.json"
    accumulated, seen_urls, resume_ts = _load_existing(out)

    time_from = resume_ts or _to_av_ts(config.TRAIN_START)
    time_to = _to_av_ts(config.NEWS_END, end_of_day=True)

    if resume_ts:
        log.info(
            "%s — resuming from %s  (%d items already saved)",
            ticker, resume_ts, len(accumulated),
        )
    else:
        log.info("%s — starting fresh  [%s → %s]", ticker, time_from, time_to)

    # Skip immediately if we already covered the full range
    if time_from >= time_to:
        log.info("%s — already complete, skipping", ticker)
        return len(accumulated)

    page = 0

    while time_from < time_to:
        page += 1
        log.info(
            "%s — page %d  fetching [%s → %s]",
            ticker, page, time_from, time_to,
        )

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": time_from,
            "time_to": time_to,
            "limit": LIMIT,
            "sort": "EARLIEST",
            "apikey": api_key,
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.error("%s — request failed (page %d): %s", ticker, page, exc)
            break

        # API-level error messages (rate limit exceeded, invalid key, etc.)
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            log.warning("%s — API message (page %d): %s", ticker, page, msg)
            break

        feed: list[dict] = data.get("feed", [])
        if not feed:
            log.info("%s — no items returned on page %d, done", ticker, page)
            break

        # Deduplicate at page boundaries (the +1 min advance handles most cases,
        # but duplicate URLs can appear when articles are re-indexed by AV).
        new_items = []
        for item in feed:
            url = item.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                new_items.append(item)

        accumulated.extend(new_items)
        _save(accumulated, out)

        log.info(
            "%s — page %d: %d returned, %d new, %d total",
            ticker, page, len(feed), len(new_items), len(accumulated),
        )

        if len(feed) < LIMIT:
            log.info("%s — last page reached (< %d items)", ticker, LIMIT)
            break

        # Advance window: next time_from = last item timestamp + 1 min
        time_from = _advance_ts(feed[-1]["time_published"])

        time.sleep(SLEEP_S)

    return len(accumulated)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    # Busca .env.development primero; si no existe, cae a .env como fallback.
    load_dotenv(dotenv_path=ROOT / ".env.development", override=False)
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        log.error(
            "ALPHAVANTAGE_API_KEY is not set — "
            "copy .env.example to .env and fill in your key"
        )
        sys.exit(1)

    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Alpha Vantage news downloader ===")
    log.info("Tickers   : %s", config.TICKERS)
    log.info("Date range: %s → %s", config.TRAIN_START, config.NEWS_END)

    summary: list[tuple] = []

    for ticker in config.TICKERS:
        total = fetch_ticker_news(ticker, api_key)

        out = NEWS_DIR / f"{ticker}_raw.json"
        earliest = latest = "N/A"
        if out.exists() and total > 0:
            items = json.loads(out.read_text())
            dates = sorted(
                it["time_published"] for it in items if "time_published" in it
            )
            if dates:
                earliest, latest = dates[0], dates[-1]

        summary.append((ticker, total, earliest, latest))
        log.info("%s — done: %d total items", ticker, total)

    # ── Final summary table ────────────────────────────────────────────────────
    col = 18
    print()
    print("=" * 76)
    print(f"  {'TICKER':<8}  {'ITEMS':>6}  {'EARLIEST':<{col}}  {'LATEST':<{col}}")
    print("-" * 76)
    for ticker, total, earliest, latest in summary:
        print(f"  {ticker:<8}  {total:>6}  {earliest:<{col}}  {latest:<{col}}")
    print("=" * 76)


if __name__ == "__main__":
    main()
