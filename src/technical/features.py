"""Technical feature engineering for OHLCV price data.

All indicators are implemented with pandas / numpy only — no ta-lib required.

Warm-up rows
------------
Rolling windows and log-return shifts create NaN values in the first N rows.
The longest lookback is 20 (Bollinger Bands, volume ratio, ret_20), so the
first 20 rows per ticker are dropped inside ``compute_features``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Feature list ───────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_pct",
    "ema_20", "ema_50", "ema_cross",
    "ret_5", "ret_10", "ret_20",
    "volume_ratio",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and append them to *df*.

    Parameters
    ----------
    df:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume``, sorted ascending by date (any DatetimeIndex or
        integer index is accepted).

    Returns
    -------
    Copy of *df* with :data:`FEATURE_COLS` appended and warm-up rows
    (where any feature is NaN) removed.
    """
    df = df.copy()
    c = df["close"]
    v = df["volume"]

    # ── RSI 14 (Wilder's smoothing = EWM with α = 1/14) ───────────────────────
    delta     = c.diff()
    gain      = delta.clip(lower=0)
    loss      = (-delta).clip(lower=0)
    avg_gain  = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss  = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs        = avg_gain / avg_loss
    df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    # ── MACD (12/26/9) ────────────────────────────────────────────────────────
    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands (window=20, 2σ) ───────────────────────────────────────
    roll          = c.rolling(20)
    df["bb_mid"]   = roll.mean()
    std20          = roll.std(ddof=1)
    df["bb_upper"] = df["bb_mid"] + 2.0 * std20
    df["bb_lower"] = df["bb_mid"] - 2.0 * std20
    band_width     = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"]   = (c - df["bb_lower"]) / band_width

    # ── EMA 20 / 50 and their ratio ────────────────────────────────────────────
    df["ema_20"]    = c.ewm(span=20, adjust=False).mean()
    df["ema_50"]    = c.ewm(span=50, adjust=False).mean()
    df["ema_cross"] = df["ema_20"] / df["ema_50"]

    # ── Log returns over 5, 10, 20 days ───────────────────────────────────────
    for n in (5, 10, 20):
        df[f"ret_{n}"] = np.log(c / c.shift(n))

    # ── Volume ratio vs 20-day mean ────────────────────────────────────────────
    df["volume_ratio"] = v / v.rolling(20).mean()

    # ── Drop warm-up rows with any NaN feature ─────────────────────────────────
    df = df.dropna(subset=FEATURE_COLS)
    return df


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    ROOT = Path(__file__).resolve().parents[2]
    path = ROOT / "data" / "raw" / "prices" / "AAPL.parquet"

    raw = pd.read_parquet(path)
    raw.columns = [c.lower() for c in raw.columns]

    result = compute_features(raw)

    print(f"Shape after feature computation: {result.shape}")
    print(f"Rows dropped (warm-up):          {len(raw) - len(result)}")
    print()
    print(result[FEATURE_COLS].head(3).to_string())
