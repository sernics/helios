"""Independent evaluation of the FinBERT sentiment module.

Two evaluations
---------------
1. **PhraseBank accuracy** — benchmark on Financial PhraseBank (all-agree
   subset) to verify that the scorer is well-calibrated before using it on
   proprietary news.

2. **Score vs. 5-day return** — check whether FinBERT sentiment on news
   aligned to a trading day is predictive of the 5-day forward return label
   in the training split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sentiment.finbert import FinBERTScorer

ROOT        = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "sentiment"

# Thresholds for converting continuous score → 3-class prediction
_POS_THRESH =  0.1
_NEG_THRESH = -0.1


# ── 1. Financial PhraseBank benchmark ─────────────────────────────────────────

def evaluate_on_phrasebank(batch_size: int = 32) -> dict:
    """Benchmark FinBERT on the Financial PhraseBank (sentences_allagree).

    Downloads the dataset from HuggingFace on first run (cached locally
    afterwards).  Labels are mapped:

        dataset label 0 (negative) → −1
        dataset label 1 (neutral)  →  0
        dataset label 2 (positive) → +1

    Returns
    -------
    dict with keys ``accuracy``, ``f1_macro``, ``confusion_matrix``.
    """
    import zipfile
    from huggingface_hub import hf_hub_download

    print("Loading Financial PhraseBank (sentences_allagree) …")

    # datasets ≥3.x dropped custom-script support.  Download the source zip
    # directly from HF Hub and parse the `Sentences_AllAgree.txt` file.
    zip_path = hf_hub_download(
        "takala/financial_phrasebank",
        "data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    _STR_TO_INT = {"negative": -1, "neutral": 0, "positive": 1}

    sentences:  list[str] = []
    raw_labels: list[int] = []

    with zipfile.ZipFile(zip_path) as zf:
        target = next(n for n in zf.namelist() if "AllAgree" in n and n.endswith(".txt"))
        text   = zf.read(target).decode("latin-1")
        for line in text.splitlines():
            line = line.strip()
            if "@" not in line:
                continue
            sentence, label_str = line.rsplit("@", 1)
            label_int = _STR_TO_INT.get(label_str.strip().lower())
            if label_int is None:
                continue
            sentences.append(sentence.strip())
            raw_labels.append(label_int)

    true_labels = raw_labels   # already mapped to {-1, 0, +1}

    print(f"  {len(sentences)} sentences — running FinBERT …")
    scorer = FinBERTScorer()

    scores: list[float] = []
    for i in range(0, len(sentences), batch_size):
        batch   = sentences[i : i + batch_size]
        results = scorer.score_batch(batch)
        scores.extend(s for s, _ in results)

    # Convert continuous score to 3-class prediction
    def _to_class(s: float) -> int:
        if s > _POS_THRESH:
            return 1
        if s < _NEG_THRESH:
            return -1
        return 0

    pred_labels = [_to_class(s) for s in scores]

    acc  = accuracy_score(true_labels, pred_labels)
    f1   = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    cm   = confusion_matrix(true_labels, pred_labels, labels=[-1, 0, 1])

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  F1 macro  : {f1:.4f}")
    print("  Confusion matrix (rows=true, cols=pred)  [neg | neu | pos]")
    print(f"    {cm}")

    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm}


# ── 2. Score vs. 5-day return ─────────────────────────────────────────────────

def evaluate_score_vs_return() -> dict:
    """Correlate FinBERT score_sent with label_5d on the training split.

    Only rows where ``conf_sent > 0`` (at least one news item was scored)
    are included.

    Returns
    -------
    dict with keys ``pearson_r``, ``pearson_p``, ``pb_r``, ``pb_p``,
    ``quintile_table``.
    """
    scores_path = RESULTS_DIR / "train_finbert_scores.csv"
    split_path  = SPLITS_DIR  / "train.parquet"

    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. Run finbert_runner.py first."
        )

    scores_df = pd.read_csv(scores_path, parse_dates=["date"])
    split_df  = pd.read_parquet(split_path)[["ticker", "date", "label_5d"]]
    split_df["date"] = pd.to_datetime(split_df["date"])

    merged = scores_df.merge(split_df, on=["ticker", "date"], how="inner")
    merged = merged[merged["conf_sent"] > 0.0].copy()
    merged["label_5d"] = merged["label_5d"].astype(int)

    print(f"  Rows with news (conf > 0): {len(merged):,}")

    s = merged["score_sent"].to_numpy()
    y = merged["label_5d"].to_numpy()

    pr, pp     = pearsonr(s, y)
    pbr, pbp   = pointbiserialr(y, s)

    print(f"\n  Pearson r            : {pr:+.4f}  (p={pp:.4f})")
    print(f"  Point-biserial r     : {pbr:+.4f}  (p={pbp:.4f})")

    # Quintile table
    merged["quintile"] = pd.qcut(
        merged["score_sent"], q=5,
        labels=["Q1\n(most bearish)", "Q2", "Q3", "Q4", "Q5\n(most bullish)"],
        duplicates="drop",
    )
    qtable = (
        merged.groupby("quintile", observed=True)["label_5d"]
        .agg(mean_label="mean", count="count")
        .reset_index()
    )
    qtable.columns = ["quintile", "mean_label_5d", "n"]

    print("\n  Score quintile → mean label_5d:")
    print(f"  {'Quintile':<22}  {'mean label_5d':>14}  {'n':>6}")
    print("  " + "-" * 46)
    for _, row in qtable.iterrows():
        print(f"  {str(row['quintile']):<22}  {row['mean_label_5d']:>14.4f}  {int(row['n']):>6}")

    return {
        "pearson_r": pr, "pearson_p": pp,
        "pb_r": pbr,     "pb_p": pbp,
        "quintile_table": qtable,
    }


# ── 3. Full report ─────────────────────────────────────────────────────────────

def run_full_evaluation() -> None:
    """Run both evaluations and print a consolidated summary."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  EVALUATION 1 — Financial PhraseBank")
    print(sep)
    pb_results = evaluate_on_phrasebank()

    print(f"\n{sep}")
    print("  EVALUATION 2 — FinBERT score vs. 5-day return (train split)")
    print(sep)
    ret_results = evaluate_score_vs_return()

    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    print(f"  PhraseBank accuracy : {pb_results['accuracy']:.4f}")
    print(f"  PhraseBank F1 macro : {pb_results['f1_macro']:.4f}")
    print(f"  Pearson r (score vs label_5d) : {ret_results['pearson_r']:+.4f}  "
          f"p={ret_results['pearson_p']:.4f}")
    print(f"  Point-biserial r              : {ret_results['pb_r']:+.4f}  "
          f"p={ret_results['pb_p']:.4f}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_evaluation()
