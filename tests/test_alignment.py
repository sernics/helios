"""Tests for src/data/alignment.py."""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.alignment import (
    align_filings_to_trading_day,
    align_news_to_trading_day,
    assert_no_leakage,
    get_trading_days,
)

ET = ZoneInfo("America/New_York")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Return an ET datetime converted to UTC."""
    return datetime(year, month, day, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def _news_df(*timestamps: datetime) -> pd.DataFrame:
    return pd.DataFrame({"time_published": list(timestamps)})


def _dummy_prices() -> pd.DataFrame:
    """Minimal prices_df; only the index matters for the alignment API."""
    days = get_trading_days("2018-01-01", "2026-12-31")
    return pd.DataFrame(index=pd.DatetimeIndex(days))


# ── Test 1: news at 09:00 ET on a Monday → same Monday ────────────────────────

def test_news_morning_same_day():
    # 2024-01-08 is a regular Monday (no holiday).
    ts = _et(2024, 1, 8, 9, 0)
    result = align_news_to_trading_day(_news_df(ts), _dummy_prices())
    assert result.iloc[0]["feature_date"] == date(2024, 1, 8)


# ── Test 2: news at 17:00 ET on Monday → Tuesday ──────────────────────────────

def test_news_afternoon_next_day():
    ts = _et(2024, 1, 8, 17, 0)   # 17:00 ET — after 16:00 close
    result = align_news_to_trading_day(_news_df(ts), _dummy_prices())
    assert result.iloc[0]["feature_date"] == date(2024, 1, 9)


# ── Test 3: news at 17:00 ET on Friday → following Monday ─────────────────────

def test_news_friday_after_close_skips_weekend():
    # 2024-01-05 is a Friday; the next trading day is Monday 2024-01-08.
    ts = _et(2024, 1, 5, 17, 0)
    result = align_news_to_trading_day(_news_df(ts), _dummy_prices())
    assert result.iloc[0]["feature_date"] == date(2024, 1, 8)


# ── Test 4: filing on Monday morning → Wednesday (T+2) ────────────────────────

def test_filing_embargo_t_plus_2():
    # Monday 09:00 ET → T = Monday.  T+2 = Wednesday.
    ts = _et(2024, 1, 8, 9, 0)
    result = align_filings_to_trading_day(_news_df(ts), _dummy_prices())
    assert result.iloc[0]["feature_date"] == date(2024, 1, 10)


# ── Test 5: assert_no_leakage passes on a clean dataframe ─────────────────────

def test_no_leakage_clean():
    # Published Sunday 20:00 ET → UTC Monday 01:00.
    # feature_date is Monday; market open is 14:30 UTC.
    # 01:00 UTC < 14:30 UTC  →  no violation.
    ts = _et(2024, 1, 7, 20, 0)
    df = pd.DataFrame({
        "time_published": [ts],
        "feature_date": [date(2024, 1, 8)],
    })
    assert_no_leakage(df, split="train")   # must not raise


# ── Test 6: assert_no_leakage raises AssertionError when given a leaking row ──

def test_no_leakage_violation():
    # Published Monday 11:00 ET = 16:00 UTC — after market open (14:30 UTC).
    ts = _et(2024, 1, 8, 11, 0)
    df = pd.DataFrame({
        "time_published": [ts],
        "feature_date": [date(2024, 1, 8)],
    })
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(df, split="test")


# ── Bonus: get_trading_days excludes weekends and NYSE holidays ────────────────

def test_get_trading_days_excludes_weekend():
    days = get_trading_days("2024-01-05", "2024-01-08")
    dates = [str(d) for d in days]
    assert "2024-01-06" not in dates   # Saturday
    assert "2024-01-07" not in dates   # Sunday
    assert "2024-01-05" in dates       # Friday
    assert "2024-01-08" in dates       # Monday


def test_get_trading_days_excludes_nyse_holiday():
    # MLK Day is 2024-01-15 — NYSE is closed.
    days = get_trading_days("2024-01-12", "2024-01-17")
    dates = [str(d) for d in days]
    assert "2024-01-15" not in dates   # MLK Day — closed
    assert "2024-01-12" in dates       # Friday before
    assert "2024-01-16" in dates       # Tuesday after
