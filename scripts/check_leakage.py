"""Verify that no split contains look-ahead bias.

For each of train / val / test:
  1. Load the parquet file from data/splits/.
  2. Explode news_items so every individual article becomes its own row,
     carrying the feature_date of the trading day it was assigned to.
  3. Call assert_no_leakage() — passes only if every article's time_published
     is strictly before 09:30 ET (market open) of its feature_date.
  4. Report ✅ or a full violation table.

Usage
-----
    python scripts/check_leakage.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.alignment import assert_no_leakage

SPLITS_DIR = ROOT / "data" / "splits"
SPLITS = ["train", "val", "test"]


def explode_news(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per news article with ``time_published`` and ``feature_date``.

    Rows with no news (news_count == 0 / news_items == '[]') are silently
    dropped — there is nothing to check for those trading days.
    """
    records = []
    for _, row in df.iterrows():
        items = json.loads(row["news_items"])
        if not items:
            continue
        fd = row["date"]
        # Normalise feature_date to datetime.date for assert_no_leakage
        if hasattr(fd, "date"):
            fd = fd.date()
        for item in items:
            records.append(
                {
                    "time_published": item["time_published"],
                    "feature_date": fd,
                }
            )

    if not records:
        return pd.DataFrame(columns=["time_published", "feature_date"])

    exploded = pd.DataFrame(records)
    exploded["time_published"] = pd.to_datetime(
        exploded["time_published"], utc=True
    )
    return exploded


def check_split(name: str) -> bool:
    """Check one split. Returns True if clean, False if violations found."""
    path = SPLITS_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)

    print(f"\n{'─' * 60}")
    print(f"  Checking split: {name.upper()}  ({len(df):,} trading-day rows)")

    exploded = explode_news(df)
    if exploded.empty:
        print(f"  ⚠️  No news items found — nothing to check.")
        return True

    print(f"  News articles to check: {len(exploded):,}")

    try:
        assert_no_leakage(exploded, split=name)
        print(f"  ✅ No leakage detected in {name}")
        return True

    except AssertionError as exc:
        print(f"  ❌ LEAKAGE DETECTED in {name}!")
        print(f"     {exc}")

        # Full violation report
        exploded["_open_utc"] = None   # assert_no_leakage already printed counts
        from zoneinfo import ZoneInfo
        from datetime import datetime, time, timezone
        ET = ZoneInfo("America/New_York")
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")

        lo = str(exploded["feature_date"].min())
        hi = str(exploded["feature_date"].max())
        schedule = nyse.schedule(start_date=lo, end_date=hi)
        open_map = {
            ts.date(): schedule.loc[ts, "market_open"].to_pydatetime()
            for ts in schedule.index
        }

        def _open_utc(d: date) -> datetime:
            if d in open_map:
                return open_map[d]
            return datetime.combine(d, time(9, 30)).replace(tzinfo=ET).astimezone(timezone.utc)

        exploded["_open_utc"] = exploded["feature_date"].map(_open_utc)
        violations = exploded[exploded["time_published"] >= exploded["_open_utc"]]

        print(f"\n  First 20 violations:")
        print(
            violations[["time_published", "feature_date", "_open_utc"]]
            .head(20)
            .to_string(index=False)
        )
        return False


def main() -> None:
    print("=" * 60)
    print("  Helios — Temporal leakage check")
    print("=" * 60)

    all_clean = True
    for name in SPLITS:
        clean = check_split(name)
        all_clean = all_clean and clean

    print(f"\n{'=' * 60}")
    if all_clean:
        print("  ✅  All splits are leakage-free.")
    else:
        print("  ❌  One or more splits have leakage — review above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
