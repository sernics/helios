"""Tests for src/technical/features.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.technical.features import FEATURE_COLS, compute_features


# ── Shared fixture ─────────────────────────────────────────────────────────────

def _sample_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV dataframe with realistic random-walk prices."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    noise = np.abs(rng.normal(0, 0.003, n))
    high   = close * (1 + noise)
    low    = close * (1 - noise)
    open_  = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    return compute_features(_sample_df())


# ── Test 1: all feature columns are present ────────────────────────────────────

def test_all_feature_columns_present(features_df):
    missing = [col for col in FEATURE_COLS if col not in features_df.columns]
    assert missing == [], f"Missing feature columns: {missing}"


# ── Test 2: no NaN values in output ───────────────────────────────────────────

def test_no_nan_in_output(features_df):
    nan_cols = [col for col in FEATURE_COLS if features_df[col].isna().any()]
    assert nan_cols == [], f"NaN values found in: {nan_cols}"


# ── Test 3: bb_pct is in [0, 1] when price is within the bands ────────────────

def test_bb_pct_within_bands(features_df):
    within = (
        (features_df["close"] >= features_df["bb_lower"]) &
        (features_df["close"] <= features_df["bb_upper"])
    )
    if within.any():
        bp = features_df.loc[within, "bb_pct"]
        assert (bp >= 0.0).all(), "bb_pct < 0 for a price within the bands"
        assert (bp <= 1.0).all(), "bb_pct > 1 for a price within the bands"


# ── Test 4: volume_ratio is always positive ────────────────────────────────────

def test_volume_ratio_always_positive(features_df):
    assert (features_df["volume_ratio"] > 0).all(), \
        "volume_ratio contains non-positive values"


# ── Test 5: ema_cross exactly equals ema_20 / ema_50 ─────────────────────────

def test_ema_cross_exact(features_df):
    expected = features_df["ema_20"] / features_df["ema_50"]
    pd.testing.assert_series_equal(
        features_df["ema_cross"],
        expected,
        check_names=False,
    )


# ── Test 6: warm-up rows are dropped (output shorter than input) ───────────────

def test_warmup_rows_dropped():
    raw = _sample_df(n=200)
    result = compute_features(raw)
    assert len(result) < len(raw), "No warm-up rows were dropped"


# ── Test 7: RSI is bounded in [0, 100] ────────────────────────────────────────

def test_rsi_bounded(features_df):
    assert (features_df["rsi_14"] >= 0).all(), "RSI < 0 found"
    assert (features_df["rsi_14"] <= 100).all(), "RSI > 100 found"


# ── Test 8: MACD histogram equals MACD minus signal ───────────────────────────

def test_macd_hist_equals_macd_minus_signal(features_df):
    expected = features_df["macd"] - features_df["macd_signal"]
    pd.testing.assert_series_equal(
        features_df["macd_hist"],
        expected,
        check_names=False,
    )
