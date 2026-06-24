"""XGBoost baseline classifier trained on technical features.

Pipeline
--------
1. Load train / val parquet splits from ``data/splits/``.
2. Compute technical features per ticker using :func:`compute_features`.
3. Train an ``XGBClassifier`` with AUC early stopping on the val set.
4. Save model to ``results/technical/xgb_model.json`` and feature
   importances to ``results/technical/feature_importances.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.technical.features import FEATURE_COLS, compute_features

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "technical"
MODEL_PATH       = RESULTS_DIR / "xgb_model_tuned.json"
IMPORTANCES_PATH = RESULTS_DIR / "feature_importances.csv"


# ── Data preparation ───────────────────────────────────────────────────────────

def _prepare_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a parquet split, compute features per ticker, return (X, y).

    Features are computed on each ticker independently so that rolling
    windows never bleed across tickers.  Warm-up rows (~20 per ticker)
    are dropped inside :func:`compute_features`.
    """
    df = pd.read_parquet(path)

    records = []
    for _ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").copy().set_index("date")
        records.append(compute_features(group))

    combined = pd.concat(records).sort_index()
    X = combined[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = combined["label_5d"].astype(int).to_numpy()
    return X, y


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    train_path: Path | None = None,
    val_path: Path | None   = None,
) -> xgb.XGBClassifier:
    """Train and save an XGBClassifier on technical features.

    Parameters
    ----------
    train_path, val_path:
        Paths to parquet split files.  Default to ``data/splits/train.parquet``
        and ``data/splits/val.parquet``.

    Returns
    -------
    Fitted ``XGBClassifier``.
    """
    train_path = train_path or SPLITS_DIR / "train.parquet"
    val_path   = val_path   or SPLITS_DIR / "val.parquet"

    print("Loading and preparing splits …")
    X_train, y_train = _prepare_split(train_path)
    X_val,   y_val   = _prepare_split(val_path)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw   = n_neg / n_pos   # scale_pos_weight = neg / pos

    print(f"  Train  {X_train.shape}  pos={n_pos}  neg={n_neg}  "
          f"scale_pos_weight={spw:.4f}")
    print(f"  Val    {X_val.shape}  pos={int(y_val.sum())}  "
          f"neg={int((y_val == 0).sum())}")

    model = xgb.XGBClassifier(
        max_depth=5,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )

    print("\nTraining …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    best_iter  = model.best_iteration
    auc_train  = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    auc_val    = roc_auc_score(y_val,   model.predict_proba(X_val)[:, 1])

    print(f"\n  Best iteration : {best_iter}")
    print(f"  AUC-ROC  train : {auc_train:.4f}")
    print(f"  AUC-ROC  val   : {auc_val:.4f}")

    # ── Feature importances ────────────────────────────────────────────────────
    importances = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS, name="importance")
        .sort_values(ascending=False)
    )

    print("\n  Top 5 features by importance:")
    for feat, imp in importances.head(5).items():
        print(f"    {feat:<20}  {imp:.4f}")

    # ── Persist ────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    importances.to_csv(IMPORTANCES_PATH, header=True)

    print(f"\n  Saved model        → {MODEL_PATH.relative_to(ROOT)}")
    print(f"  Saved importances  → {IMPORTANCES_PATH.relative_to(ROOT)}")

    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def load_model() -> xgb.XGBClassifier:
    """Load and return the trained model from ``results/technical/xgb_model.json``."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/technical/model.py` first."
        )
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
