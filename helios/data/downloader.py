"""Download OHLCV price data and news for all configured tickers."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# Allow `python helios/data/downloader.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helios import config  # noqa: E402  (import after sys.path mutation)

ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data" / "raw" / "prices"
NEWS_DIR = ROOT / "data" / "raw" / "news"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_gaps(index: pd.DatetimeIndex) -> list:
    """Return dates after which a gap of >7 calendar days exists.

    Single and double US market holidays (MLK Day, Presidents' Day, July 4th,
    Thanksgiving, Christmas) create at most 4-day calendar gaps and are ignored.
    A gap exceeding a full calendar week indicates genuinely missing data.
    """
    if len(index) < 2:
        return []
    diffs = pd.Series(index.to_list()).diff().dropna()
    flagged = index[1:][diffs > pd.Timedelta(days=7)]
    return [str(d.date()) for d in flagged]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ticker level from a MultiIndex returned by yfinance ≥0.2."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def _normalize_news_item(item: dict) -> dict:
    """Ensure every news record has a top-level ``providerPublishTime`` (Unix int).

    yfinance ≥1.x wraps items as ``{"id": …, "content": {…, "pubDate": "…Z"}}``.
    Older versions exposed ``providerPublishTime`` directly as an integer.
    Both representations are preserved in the saved JSON; only the timestamp
    field is added when absent.
    """
    record = dict(item)

    if "providerPublishTime" not in record:
        pubdate: str | None = (record.get("content") or {}).get("pubDate")
        if pubdate:
            try:
                dt = datetime.fromisoformat(pubdate.replace("Z", "+00:00"))
                record["providerPublishTime"] = int(dt.timestamp())
            except (ValueError, AttributeError):
                record["providerPublishTime"] = None
        else:
            record["providerPublishTime"] = None

    return record


# ── Downloaders ────────────────────────────────────────────────────────────────

def download_prices() -> None:
    """Download daily OHLCV for every ticker and persist as Parquet."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in config.TICKERS:
        log.info("Downloading prices: %s", ticker)
        try:
            df = yf.download(
                ticker,
                start=config.TRAIN_START,
                end=config.TEST_END,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            log.error("%s — download failed: %s", ticker, exc)
            continue

        if df.empty:
            log.warning("%s — no data returned", ticker)
            continue

        df = _flatten_columns(df)

        rows = len(df)
        date_min = df.index.min().date()
        date_max = df.index.max().date()
        log.info("%s — %d rows  [%s → %s]", ticker, rows, date_min, date_max)

        gaps = _detect_gaps(df.index)
        if gaps:
            log.warning(
                "%s — %d potential gap(s) detected (session after gap): %s",
                ticker,
                len(gaps),
                gaps,
            )
        else:
            log.info("%s — no date gaps detected", ticker)

        out = PRICES_DIR / f"{ticker}.parquet"
        df.to_parquet(out)
        log.info("%s — saved → %s", ticker, out.relative_to(ROOT))


def download_news() -> None:
    """Fetch available news for every ticker and persist as JSON."""
    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in config.TICKERS:
        log.info("Fetching news: %s", ticker)
        try:
            raw_items: list[dict] = yf.Ticker(ticker).news
        except Exception as exc:
            log.error("%s — news fetch failed: %s", ticker, exc)
            continue

        if not raw_items:
            log.warning("%s — no news items returned", ticker)
            continue

        cleaned = [_normalize_news_item(item) for item in raw_items]

        missing_ts = sum(1 for r in cleaned if r.get("providerPublishTime") is None)
        if missing_ts:
            log.warning(
                "%s — %d/%d items missing providerPublishTime",
                ticker,
                missing_ts,
                len(cleaned),
            )

        out = NEWS_DIR / f"{ticker}.json"
        out.write_text(json.dumps(cleaned, indent=2, default=str))
        log.info(
            "%s — %d news items saved → %s", ticker, len(cleaned), out.relative_to(ROOT)
        )


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Helios data downloader ===")
    log.info("Tickers   : %s", config.TICKERS)
    log.info("Date range: %s → %s", config.TRAIN_START, config.TEST_END)

    download_prices()
    download_news()

    log.info("=== Download complete ===")


if __name__ == "__main__":
    main()
