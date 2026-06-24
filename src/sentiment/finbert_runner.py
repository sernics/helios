"""Run FinBERT over all downloaded news and produce daily sentiment scores.

For each row in a data split (ticker × trading-day) the runner:

1. Parses the ``news_items`` JSON column.
2. Builds one text string per article as ``title + ". " + summary``.
3. Scores all texts for the split in global batches of 32 (single model load,
   much faster than per-row loading).
4. Maps scores back to rows and aggregates with ``aggregate_daily()``.

Results are cached in ``results/sentiment/{split}_finbert_scores.csv``.
Re-running skips splits whose cache file already exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sentiment.aggregator import aggregate_daily
from src.sentiment.finbert import FinBERTScorer

ROOT        = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "sentiment"
BATCH_SIZE  = 32


# ── Per-split runner ───────────────────────────────────────────────────────────

def run_finbert_on_split(
    split: str,
    scorer: FinBERTScorer | None = None,
) -> pd.DataFrame:
    """Score every row in a split and save to CSV.

    Parameters
    ----------
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    scorer:
        Optional pre-loaded ``FinBERTScorer``.  When *None*, a new scorer is
        created (model downloaded / loaded from HF cache).

    Returns
    -------
    DataFrame with columns ``ticker, date, score_sent, conf_sent``.
    """
    out_path = RESULTS_DIR / f"{split}_finbert_scores.csv"

    if out_path.exists():
        print(f"[{split}] Cache hit — loading {out_path.relative_to(ROOT)}")
        return pd.read_csv(out_path, parse_dates=["date"])

    df = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")

    if scorer is None:
        scorer = FinBERTScorer()

    # ── Build a flat list of all texts with row-index mapping ──────────────────
    row_slices: list[tuple[int, int]] = []   # (start, end) into flat_texts
    flat_texts: list[str] = []

    for _, row in df.iterrows():
        start = len(flat_texts)
        if row["news_count"] > 0:
            try:
                items = json.loads(row["news_items"])
            except (json.JSONDecodeError, TypeError):
                items = []
            for item in items:
                title   = item.get("title",   "") or ""
                summary = item.get("summary", "") or ""
                flat_texts.append(f"{title}. {summary}".strip())
        row_slices.append((start, len(flat_texts)))

    # ── Batch FinBERT inference ────────────────────────────────────────────────
    flat_scores: list[tuple[float, float]] = []
    n_batches = (len(flat_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(n_batches), desc=f"[{split}] FinBERT", unit="batch"):
        batch = flat_texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        flat_scores.extend(scorer.score_batch(batch))

    # ── Aggregate per row ──────────────────────────────────────────────────────
    records = []
    for idx, (start, end) in enumerate(row_slices):
        row      = df.iloc[idx]
        pairs    = flat_scores[start:end]
        s, c     = aggregate_daily(pairs)
        records.append({
            "ticker":     row["ticker"],
            "date":       row["date"],
            "score_sent": s,
            "conf_sent":  c,
        })

    result = pd.DataFrame(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"[{split}] Saved → {out_path.relative_to(ROOT)}")

    return result


# ── Summary helper ─────────────────────────────────────────────────────────────

def _print_summary(split: str, scores_df: pd.DataFrame, split_df: pd.DataFrame) -> None:
    total_rows   = len(scores_df)
    has_news     = int((split_df["news_count"] > 0).sum())
    s            = scores_df["score_sent"]
    c            = scores_df["conf_sent"]
    high_conf    = (c > 0.2).mean() * 100

    print(
        f"\n[{split:5s}] "
        f"rows={total_rows:>6d}  has_news={has_news:>6d}  "
        f"score mean={s.mean():+.4f}  std={s.std():.4f}  "
        f"conf mean={c.mean():.4f}  std={c.std():.4f}  "
        f"conf>0.2: {high_conf:.1f}%"
    )


# ── Run all splits ─────────────────────────────────────────────────────────────

def run_all_splits() -> None:
    """Score all three splits, reusing the same FinBERT model instance."""
    scorer = FinBERTScorer()
    print(f"FinBERT device: {scorer._device}\n")

    for split in ("train", "val", "test"):
        scores_df = run_finbert_on_split(split, scorer=scorer)
        split_df  = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")
        _print_summary(split, scores_df, split_df)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all_splits()
