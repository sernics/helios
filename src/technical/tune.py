"""Optuna hyperparameter search for the XGBoost technical model.

Flow
----
1. Features are computed once from the parquet splits (outside the search loop).
2. Each Optuna trial trains an ``XGBClassifier`` on those arrays and returns
   AUC-ROC on the val set.
3. After *n_trials*, the best params are used to retrain a final model on the
   combined train+val data (no early stopping, using the best iteration count
   found during search).
4. Final model is saved to ``results/technical/xgb_model_tuned.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.technical.features import FEATURE_COLS, compute_features
from src.technical.model import RESULTS_DIR, SPLITS_DIR

ROOT = Path(__file__).resolve().parents[2]

TUNED_MODEL_PATH = RESULTS_DIR / "xgb_model_tuned.json"
BEST_PARAMS_PATH = RESULTS_DIR / "best_params.json"

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _df_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute features per ticker and return (X, y) numpy arrays."""
    records = []
    for _ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").copy().set_index("date")
        records.append(compute_features(group))
    combined = pd.concat(records).sort_index()
    X = combined[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = combined["label_5d"].astype(int).to_numpy()
    return X, y


def _spw(y: np.ndarray) -> float:
    """scale_pos_weight = neg / pos."""
    pos = int(y.sum())
    neg = len(y) - pos
    return neg / pos if pos > 0 else 1.0


# ── Objective ──────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """Optuna objective: suggest params, train XGB, return val AUC.

    Parameters
    ----------
    trial:
        Current Optuna trial.
    train_df, val_df:
        Raw split DataFrames (OHLCV + ``label_5d``).  Features are computed
        inside this function so the signature stays self-contained.

    Returns
    -------
    AUC-ROC on the validation set.
    """
    X_train, y_train = _df_to_arrays(train_df)
    X_val,   y_val   = _df_to_arrays(val_df)

    params = dict(
        max_depth        = trial.suggest_int  ("max_depth",        3,    7),
        n_estimators     = trial.suggest_int  ("n_estimators",   100,  500),
        learning_rate    = trial.suggest_float("learning_rate",  0.01, 0.20, log=True),
        subsample        = trial.suggest_float("subsample",      0.6,  1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight = trial.suggest_int  ("min_child_weight",  1,   10),
        gamma            = trial.suggest_float("gamma",          0.0,  1.0),
    )

    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight  = _spw(y_train),
        eval_metric       = "auc",
        early_stopping_rounds = 20,
        random_state      = 42,
        verbosity         = 0,
    )
    model.fit(
        X_train, y_train,
        eval_set  = [(X_val, y_val)],
        verbose   = False,
    )

    # store best_iteration so run_tuning can use it for the final model
    trial.set_user_attr("best_n_estimators", model.best_iteration + 1)

    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    return auc


# ── Study ──────────────────────────────────────────────────────────────────────

def run_tuning(n_trials: int = 50) -> dict:
    """Run Optuna study, retrain final model, persist artifacts.

    Parameters
    ----------
    n_trials:
        Number of Optuna trials (default 50).

    Returns
    -------
    Best hyperparameters as a plain dict.
    """
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    # Pre-compute feature arrays once so the objective can be called cheaply
    # via a thin closure when Optuna only needs (trial,) signature.
    X_train, y_train = _df_to_arrays(train_df)
    X_val,   y_val   = _df_to_arrays(val_df)

    def _fast_objective(trial: optuna.Trial) -> float:
        """Same logic as objective() but uses pre-built arrays."""
        params = dict(
            max_depth        = trial.suggest_int  ("max_depth",        3,    7),
            n_estimators     = trial.suggest_int  ("n_estimators",   100,  500),
            learning_rate    = trial.suggest_float("learning_rate",  0.01, 0.20, log=True),
            subsample        = trial.suggest_float("subsample",      0.6,  1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight = trial.suggest_int  ("min_child_weight",  1,   10),
            gamma            = trial.suggest_float("gamma",          0.0,  1.0),
        )
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight      = _spw(y_train),
            eval_metric           = "auc",
            early_stopping_rounds = 20,
            random_state          = 42,
            verbosity             = 0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        trial.set_user_attr("best_n_estimators", model.best_iteration + 1)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    print(f"Running Optuna study — {n_trials} trials …")
    study = optuna.create_study(direction="maximize")
    study.optimize(_fast_objective, n_trials=n_trials, show_progress_bar=True)

    best        = study.best_trial
    best_params = best.params
    best_auc    = best.value
    best_n_est  = best.user_attrs.get("best_n_estimators", best_params["n_estimators"])

    print(f"\nBest trial AUC  : {best_auc:.4f}")
    print("Best params     :")
    for k, v in best_params.items():
        print(f"  {k:<22} {v}")

    # ── Final model on train+val combined ──────────────────────────────────────
    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    print(f"\nRetraining final model on train+val  "
          f"({len(X_all)} rows, n_estimators={best_n_est}) …")

    final_model = xgb.XGBClassifier(
        **{**best_params, "n_estimators": best_n_est},
        scale_pos_weight = _spw(y_all),
        eval_metric      = "auc",
        random_state     = 42,
        verbosity        = 0,
    )
    final_model.fit(X_all, y_all, verbose=False)

    val_auc_final = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])
    print(f"Final model AUC on val : {val_auc_final:.4f}")

    # ── Persist ────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    final_model.save_model(str(TUNED_MODEL_PATH))

    params_to_save = {**best_params, "n_estimators": best_n_est, "best_trial_auc": best_auc}
    BEST_PARAMS_PATH.write_text(json.dumps(params_to_save, indent=2))

    print(f"\nSaved tuned model  → {TUNED_MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved best params  → {BEST_PARAMS_PATH.relative_to(ROOT)}")

    return best_params


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_tuning(n_trials=50)
