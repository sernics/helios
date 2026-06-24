"""Score mapper for the technical XGBoost model.

Converts raw ``predict_proba`` output into a bounded signal and confidence:

    score      = 2 * p_up - 1   →  range [-1, +1]
    confidence = |score|         →  range [ 0,  1]

A positive score means the model leans bullish; negative means bearish.
Confidence captures how far the prediction is from the 0.5 indecision line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.technical.features import FEATURE_COLS, compute_features
from src.technical.model import RESULTS_DIR, SPLITS_DIR, load_model

ROOT = Path(__file__).resolve().parents[2]


# ── Core signal transform ──────────────────────────────────────────────────────

def predict_score(
    model: xgb.XGBClassifier,
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map model probabilities to a bounded score and confidence.

    Parameters
    ----------
    model:
        A fitted ``XGBClassifier``.
    X:
        Feature matrix — shape ``(n_samples, n_features)``.

    Returns
    -------
    score : np.ndarray, shape (n_samples,)
        Signal in ``[-1, 1]``.  Positive → bullish, negative → bearish.
    confidence : np.ndarray, shape (n_samples,)
        Absolute conviction in ``[0, 1]``.
    """
    p_up       = model.predict_proba(X)[:, 1]
    score      = 2.0 * p_up - 1.0
    confidence = np.abs(score)
    return score, confidence


# ── Split-level scoring ────────────────────────────────────────────────────────

def predict_score_for_split(split: str) -> pd.DataFrame:
    """Score every row in a data split and save the results.

    Parameters
    ----------
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.

    Returns
    -------
    DataFrame with columns ``ticker, date, score_tech, conf_tech, label_5d``.
    The file is also written to ``results/technical/{split}_scores.csv``.
    """
    split_path = SPLITS_DIR / f"{split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    df    = pd.read_parquet(split_path)
    model = load_model()

    records = []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").copy().set_index("date")
        group = compute_features(group)          # adds FEATURE_COLS, drops NaN rows

        X               = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        score, conf     = predict_score(model, X)

        chunk = pd.DataFrame({
            "ticker":    ticker,
            "date":      group.index,
            "score_tech": score,
            "conf_tech":  conf,
            "label_5d":  group["label_5d"].astype(int).to_numpy(),
        })
        records.append(chunk)

    result = pd.concat(records, ignore_index=True).sort_values(["date", "ticker"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{split}_scores.csv"
    result.to_csv(out_path, index=False)
    print(f"  Saved {split} scores → {out_path.relative_to(ROOT)}")

    return result


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for split_name in ("train", "val", "test"):
        scores = predict_score_for_split(split_name)

        s      = scores["score_tech"]
        high_conf_pct = (scores["conf_tech"] > 0.2).mean() * 100

        print(
            f"\n[{split_name:5s}] rows={len(scores):>5d}  "
            f"score  mean={s.mean():+.4f}  std={s.std():.4f}  "
            f"min={s.min():+.4f}  max={s.max():+.4f}  "
            f"conf>0.2: {high_conf_pct:.1f}%"
        )
