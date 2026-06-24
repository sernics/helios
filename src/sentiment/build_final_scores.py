"""Merge FinBERT CSV scores with split parquets and save sentiment parquets.

Output files (data/splits/):
    train_sentiment.parquet
    val_sentiment.parquet
    test_sentiment.parquet

Columns: ticker, date, score_sent, conf_sent, label_5d
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT        = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "splits"
SENT_DIR    = ROOT / "results" / "sentiment"
KEEP_COLS   = ["ticker", "date", "score_sent", "conf_sent", "label_5d"]


def build_sentiment_split(split: str) -> pd.DataFrame:
    scores_path = SENT_DIR  / f"{split}_finbert_scores.csv"
    parquet_path = SPLITS_DIR / f"{split}.parquet"
    out_path    = SPLITS_DIR  / f"{split}_sentiment.parquet"

    scores_df = pd.read_csv(scores_path, parse_dates=["date"])
    split_df  = pd.read_parquet(parquet_path, columns=["ticker", "date", "label_5d"])
    split_df["date"] = pd.to_datetime(split_df["date"])

    merged = scores_df.merge(split_df, on=["ticker", "date"], how="inner")
    merged["label_5d"] = merged["label_5d"].astype(int)
    merged = merged[KEEP_COLS].sort_values(["date", "ticker"]).reset_index(drop=True)

    merged.to_parquet(out_path, index=False)
    return merged


def build_all() -> None:
    for split in ("train", "val", "test"):
        df = build_sentiment_split(split)

        nulls = df.isnull().sum().sum()
        print(
            f"[{split:5s}]  rows={len(df):>6d}  nulls={nulls}  "
            f"score=[{df['score_sent'].min():+.4f}, {df['score_sent'].max():+.4f}]  "
            f"conf=[{df['conf_sent'].min():.4f}, {df['conf_sent'].max():.4f}]  "
            f"saved → {split}_sentiment.parquet"
        )


if __name__ == "__main__":
    build_all()
