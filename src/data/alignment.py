"""Temporal alignment utilities for the Helios pipeline.

All published timestamps are UTC.  The NYSE open (09:30 ET) is the cutoff
that determines which trading day a piece of information first becomes
usable as a feature — only pre-market articles are safe to assign to day T:

    published < 09:30 ET  →  feature_date = T          (pre-market, safe)
    published ≥ 09:30 ET  →  feature_date = T+1         (market is open / post-close)

Using the 16:00 ET close as the cutoff would assign intraday articles to T,
but those articles can reflect same-day price movements — genuine leakage.
The 09:30 ET cutoff guarantees that every feature_date row contains only
information that was available *before* the market opened on that day.

Filings carry an additional 24 h embargo:

    feature_date = T+2  (two trading days after the publication day)

``feature_date`` is always a valid NYSE trading day — weekends and market
holidays are automatically skipped.
"""
from __future__ import annotations

import bisect
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ET = ZoneInfo("America/New_York")
_CLOSE = time(16, 0)   # NYSE hard close — cutoff for same-day assignment
_OPEN  = time(9, 30)   # NYSE open — used in the leakage guard

_nyse = mcal.get_calendar("NYSE")


# ── Public calendar helper ─────────────────────────────────────────────────────

def get_trading_days(start_date: str, end_date: str) -> list:
    """Return a sorted list of NYSE trading days (``datetime.date``) between
    *start_date* and *end_date*, inclusive."""
    schedule = _nyse.schedule(start_date=start_date, end_date=end_date)
    return [ts.date() for ts in schedule.index]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_sorted_days(lo: date, hi: date) -> list:
    """Trading days with a ±14-day buffer around the data window."""
    return get_trading_days(
        str(lo - timedelta(days=14)),
        str(hi + timedelta(days=14)),
    )


def _next_from(candidate: date, sorted_days: list) -> date:
    """First trading day >= *candidate*."""
    idx = bisect.bisect_left(sorted_days, candidate)
    if idx >= len(sorted_days):
        raise ValueError(f"No trading day found on or after {candidate}")
    return sorted_days[idx]


def _advance(base: date, n: int, sorted_days: list) -> date:
    """Trading day *n* positions after *base* (base = position 0)."""
    idx = bisect.bisect_left(sorted_days, base)
    target = idx + n
    if target >= len(sorted_days):
        raise ValueError(f"Not enough trading days to advance {n} from {base}")
    return sorted_days[target]


def _news_feature_date(pub_utc: datetime, sorted_days: list) -> date:
    """Assign feature_date using the 09:30 ET (market open) cutoff.

    Only articles published *before* the market opens on day T can be
    attributed to T without look-ahead bias.  Anything published at or
    after 09:30 ET (intraday or post-close) is pushed to T+1.
    """
    pub_et = pub_utc.astimezone(ET)
    if pub_et.time() < _OPEN:
        candidate = pub_et.date()
    else:
        candidate = pub_et.date() + timedelta(days=1)
    return _next_from(candidate, sorted_days)


def _filing_feature_date(pub_utc: datetime, sorted_days: list) -> date:
    """Same 16:00 ET cutoff to determine T, then advance two trading days."""
    t = _news_feature_date(pub_utc, sorted_days)
    return _advance(t, 2, sorted_days)


# ── Public alignment functions ─────────────────────────────────────────────────

def align_news_to_trading_day(
    news_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign a ``feature_date`` to each news item using the 16:00 ET cutoff.

    Parameters
    ----------
    news_df:
        Must contain a ``time_published`` column (UTC-aware datetime or
        any value coercible by ``pd.to_datetime(..., utc=True)``).
    prices_df:
        DataFrame whose index contains the valid trading days for this
        ticker.  Used only as a hint; the NYSE calendar is authoritative.

    Returns
    -------
    Copy of *news_df* with a new ``feature_date`` column (``datetime.date``).
    """
    df = news_df.copy()
    df["time_published"] = pd.to_datetime(df["time_published"], utc=True)

    lo = df["time_published"].min().date()
    hi = df["time_published"].max().date()
    sorted_days = _build_sorted_days(lo, hi)

    df["feature_date"] = df["time_published"].map(
        lambda ts: _news_feature_date(ts, sorted_days)
    )
    return df


def align_filings_to_trading_day(
    filings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign a ``feature_date`` to each filing with a T+2 embargo.

    The 16:00 ET cutoff first determines T (same rule as news); the feature
    date is then set two trading days later to respect the 24 h SEC embargo.

    Parameters
    ----------
    filings_df:
        Must contain a ``time_published`` column (UTC-aware datetime).
    prices_df:
        Same role as in :func:`align_news_to_trading_day`.

    Returns
    -------
    Copy of *filings_df* with a new ``feature_date`` column.
    """
    df = filings_df.copy()
    df["time_published"] = pd.to_datetime(df["time_published"], utc=True)

    lo = df["time_published"].min().date()
    hi = df["time_published"].max().date()
    sorted_days = _build_sorted_days(lo, hi)

    df["feature_date"] = df["time_published"].map(
        lambda ts: _filing_feature_date(ts, sorted_days)
    )
    return df


# ── Leakage guard ──────────────────────────────────────────────────────────────

def assert_no_leakage(features_df: pd.DataFrame, split: str) -> None:
    """Assert that no row in *features_df* uses information from the future.

    A row is a leak if its ``time_published`` is **not** strictly before the
    NYSE market open (09:30 ET) of its ``feature_date``.  The market open
    time is read from the official NYSE schedule so early-close days are
    handled correctly.

    Parameters
    ----------
    features_df:
        Must contain columns:
        - ``feature_date``: ``datetime.date``
        - ``time_published``: UTC-aware datetime (or coercible string/int)
    split:
        Label for reporting (e.g. ``'train'``, ``'val'``, ``'test'``).

    Raises
    ------
    AssertionError
        If any violation is found, with a message showing the first offender.
    """
    df = features_df.copy()
    df["time_published"] = pd.to_datetime(df["time_published"], utc=True)

    # Build a map: feature_date → market_open UTC from the NYSE schedule
    lo = df["feature_date"].min()
    hi = df["feature_date"].max()
    if isinstance(lo, date) and not isinstance(lo, datetime):
        lo_str, hi_str = str(lo), str(hi)
    else:
        lo_str = str(lo.date()) if hasattr(lo, "date") else str(lo)
        hi_str = str(hi.date()) if hasattr(hi, "date") else str(hi)

    schedule = _nyse.schedule(start_date=lo_str, end_date=hi_str)
    open_map = {ts.date(): schedule.loc[ts, "market_open"] for ts in schedule.index}

    def _open_utc(d: date) -> datetime:
        if d in open_map:
            # market_open is already UTC-aware from pandas_market_calendars
            return open_map[d].to_pydatetime()
        # Fallback for non-trading days (shouldn't happen if alignment is correct)
        return datetime.combine(d, _OPEN).replace(tzinfo=ET).astimezone(timezone.utc)

    df["_open_utc"] = df["feature_date"].map(_open_utc)
    violations = df[df["time_published"] >= df["_open_utc"]]

    n_total = len(df)
    n_bad = len(violations)
    print(f"[{split}] leakage check — {n_total:,} rows checked, {n_bad} violation(s)")

    if n_bad:
        row = violations.iloc[0]
        raise AssertionError(
            f"Data leakage in split '{split}': "
            f"time_published={row['time_published']} is not strictly before "
            f"market open={row['_open_utc']} of feature_date={row['feature_date']}. "
            f"({n_bad}/{n_total} violation(s))"
        )
