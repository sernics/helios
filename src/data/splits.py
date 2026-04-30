"""Build train / val / test splits with OHLCV, temporal-aligned news, and labels.

Output schema per row
---------------------
ticker          str
date            datetime64[ns]
open/high/low/close/volume   float64
label_5d        int   (1 if close[t+5] > close[t] else 0; last 5 rows dropped)
news_items      str   (JSON-encoded list of news dicts for this feature_date)
news_count      int

Six parquet files are written to data/splits/:
  train.parquet, val.parquet, test.parquet
  train_norm.parquet, val_norm.parquet, test_norm.parquet

Plus data/splits/normalization_params.json (mean/std computed on train only).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.data.alignment import align_news_to_trading_day

ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data" / "raw" / "prices"
NEWS_DIR   = ROOT / "data" / "raw" / "news"
SPLITS_DIR = ROOT / "data" / "splits"

NUMERIC_COLS = ["open", "high", "low", "close", "volume"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_prices(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(PRICES_DIR / f"{ticker}.parquet")
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df


def _load_and_align_news(ticker: str, prices_df: pd.DataFrame) -> dict:
    """Return ``{datetime.date: json_str}`` mapping feature_date → news list.

    The news list is JSON-encoded so it can be stored verbatim in a parquet
    object column without pyarrow type-inference issues.
    """
    path = NEWS_DIR / f"{ticker}_raw.json"
    if not path.exists():
        log.warning("%s — news file not found, skipping", ticker)
        return {}

    items = json.loads(path.read_text())
    if not items:
        return {}

    news_df = pd.DataFrame(items)
    if "time_published" not in news_df.columns:
        return {}

    log.info("%s — aligning %d news items to trading days", ticker, len(news_df))
    aligned = align_news_to_trading_day(news_df, prices_df)

    grouped: dict = {}
    for feature_date, grp in aligned.groupby("feature_date"):
        records = grp.drop(columns=["feature_date"]).to_dict(orient="records")
        grouped[feature_date] = json.dumps(records, default=str)

    return grouped


# ── Per-ticker builder ─────────────────────────────────────────────────────────

def _build_ticker_df(ticker: str) -> pd.DataFrame:
    prices = _load_prices(ticker)

    # ── label_5d ────────────────────────────────────────────────────────────────
    # shift(-5): row t gets the close value at t+5; last 5 rows become NaN.
    prices["label_5d"] = (prices["close"].shift(-5) > prices["close"]).astype("Int64")
    prices = prices.iloc[:-5].copy()           # drop rows without a future close

    # ── news alignment ──────────────────────────────────────────────────────────
    news_by_date = _load_and_align_news(ticker, prices)

    prices["news_items"] = [
        news_by_date.get(ts.date(), "[]")
        for ts in prices.index
    ]
    prices["news_count"] = prices["news_items"].map(
        lambda s: len(json.loads(s))
    )

    prices["ticker"] = ticker
    prices = prices.reset_index()             # date becomes a regular column
    log.info("%s — %d rows built", ticker, len(prices))
    return prices


# ── Main builder ───────────────────────────────────────────────────────────────

def build_splits() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Build per-ticker DataFrames ─────────────────────────────────────────
    dfs = [_build_ticker_df(t) for t in tqdm(config.TICKERS, desc="Tickers")]
    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    log.info("Combined: %d rows across %d tickers", len(combined), len(config.TICKERS))

    # ── 2. Date-based splits ───────────────────────────────────────────────────
    def _slice(start: str, end: str) -> pd.DataFrame:
        return combined[
            (combined["date"] >= start) & (combined["date"] <= end)
        ].copy()

    train = _slice(config.TRAIN_START, config.TRAIN_END)
    val   = _slice(config.VAL_START,   config.VAL_END)
    test  = _slice(config.TEST_START,  config.TEST_END)

    log.info(
        "Splits — train:%d  val:%d  test:%d",
        len(train), len(val), len(test),
    )

    # ── 3. Normalisation params (train only) ───────────────────────────────────
    params: dict = {}
    for col in NUMERIC_COLS:
        mu  = float(train[col].mean())
        std = float(train[col].std())
        params[col] = {"mean": mu, "std": max(std, 1e-8)}

    norm_path = SPLITS_DIR / "normalization_params.json"
    norm_path.write_text(json.dumps(params, indent=2))
    log.info("Saved normalisation params → %s", norm_path.relative_to(ROOT))

    # ── 4. Apply normalisation ─────────────────────────────────────────────────
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in NUMERIC_COLS:
            out[col] = (out[col] - params[col]["mean"]) / params[col]["std"]
        return out

    train_norm = _normalise(train)
    val_norm   = _normalise(val)
    test_norm  = _normalise(test)

    # ── 5. Save all six parquet files ──────────────────────────────────────────
    to_save = [
        (train,      "train"),
        (val,        "val"),
        (test,       "test"),
        (train_norm, "train_norm"),
        (val_norm,   "val_norm"),
        (test_norm,  "test_norm"),
    ]
    for df, name in to_save:
        path = SPLITS_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("Saved %s → %s  (%d rows)", name, path.relative_to(ROOT), len(df))

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  {'SPLIT':<12} {'ROWS':>7}  {'AVG NEWS/DAY':>13}  {'LABEL=1 (%)':>12}")
    print("-" * 72)
    for df, name in to_save[:3]:
        rows        = len(df)
        avg_news    = df["news_count"].mean()
        pct_pos     = df["label_5d"].mean() * 100
        print(f"  {name:<12} {rows:>7,}  {avg_news:>13.2f}  {pct_pos:>11.1f}%")

    print()
    print("  Rows per ticker per split:")
    for df, name in to_save[:3]:
        per_ticker = df.groupby("ticker").size().to_dict()
        ticker_str = "  ".join(f"{t}:{n}" for t, n in sorted(per_ticker.items()))
        print(f"  {name:<10}  {ticker_str}")

    print("=" * 72)
    print()

    # Confirm all 6 files exist
    for _, name in to_save:
        p = SPLITS_DIR / f"{name}.parquet"
        size_mb = p.stat().st_size / 1e6
        print(f"  ✓  {p.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    p = norm_path
    print(f"  ✓  {p.relative_to(ROOT)}")
    print()


if __name__ == "__main__":
    build_splits()
