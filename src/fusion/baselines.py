"""Baseline trading strategies for the fusion evaluation suite.

B1 — Buy-and-hold          : buy equal weight on day 1, hold to end.
B2 — Threshold             : rule-based long/short from a single signal.
B3 — Early-fusion (stub)   : placeholder until PPO is trained.

All strategies return a ``pd.Series`` of daily portfolio value (starts at 1.0),
indexed by ``date``.  ``compute_metrics`` reduces that series to a standard
performance dict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

ROOT       = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
TECH_DIR   = ROOT / "results" / "technical"
SENT_DIR   = ROOT / "results" / "sentiment"
FUSION_DIR = ROOT / "results" / "fusion"

load_dotenv(dotenv_path=ROOT / ".env.development", override=False)
load_dotenv(dotenv_path=ROOT / ".env", override=False)

_THRESH_GRID = [0.005, 0.010, 0.015, 0.020, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_prices(split: str) -> pd.DataFrame:
    """Return (ticker, date, close) sorted by ticker → date."""
    df = pd.read_parquet(
        SPLITS_DIR / f"{split}.parquet", columns=["ticker", "date", "close"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_scores(split: str, signal: str) -> pd.DataFrame:
    """Return (ticker, date, score) for the requested signal."""
    if signal == "tech":
        df = pd.read_csv(TECH_DIR / f"{split}_scores.csv", parse_dates=["date"])
        return df[["ticker", "date", "score_tech"]].rename(columns={"score_tech": "score"})
    if signal == "sent":
        df = pd.read_csv(SENT_DIR / f"{split}_finbert_scores.csv", parse_dates=["date"])
        return df[["ticker", "date", "score_sent"]].rename(columns={"score_sent": "score"})
    raise ValueError(f"Unknown signal '{signal}'. Use 'tech' or 'sent'.")


def _portfolio_value(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    lambda_cost: float,
) -> pd.Series:
    """Compute equal-weight portfolio value series from per-ticker positions.

    Parameters
    ----------
    prices:
        DataFrame with columns ``ticker, date, close``.
    positions:
        DataFrame with columns ``ticker, date, position`` (−1, 0, or +1).
    lambda_cost:
        Cost charged per position change (regardless of direction).

    Returns
    -------
    ``pd.Series`` of daily portfolio value indexed by ``date`` (starts at 1.0).
    """
    df = prices.merge(positions, on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["position"] = df["position"].fillna(0.0)

    # Daily simple return per ticker
    df["ret"] = df.groupby("ticker")["close"].pct_change().fillna(0.0)

    # Detect position changes → transaction cost
    df["pos_prev"] = df.groupby("ticker")["position"].shift(1).fillna(0.0)
    df["cost"]     = (df["position"] != df["pos_prev"]).astype(float) * lambda_cost

    # P&L for day t = position entered at prior close × today's return − cost
    df["pnl"] = df["pos_prev"] * df["ret"] - df["cost"]

    # Equal-weight: average across tickers for each date
    daily = df.groupby("date")["pnl"].mean().sort_index()
    pv    = (1.0 + daily).cumprod()
    pv.iloc[0] = (1.0 + daily.iloc[0])
    return pv


# ── B1 — Buy and hold ──────────────────────────────────────────────────────────

def baseline_buy_and_hold(split: str = "test") -> pd.Series:
    """Buy all tickers on day 1, hold until the last day (equal weight).

    Returns
    -------
    Daily portfolio value ``pd.Series`` indexed by ``date``.
    """
    prices = _load_prices(split)
    dates  = prices["date"].sort_values().unique()

    # Position = 1 (long) for every ticker on every date
    pos_df = prices[["ticker", "date"]].copy()
    pos_df["position"] = 1.0

    return _portfolio_value(prices, pos_df, lambda_cost=config.LAMBDA_COST)


# ── B2 — Threshold rule ────────────────────────────────────────────────────────

def baseline_threshold(
    split: str = "val",
    signal: str = "tech",
    threshold: float = 0.2,
) -> pd.Series:
    """Signal-threshold long/short strategy.

    Rules (applied independently per ticker):

    * ``score > threshold``  → position = +1 (long)
    * ``score < −threshold`` → position = −1 (short)
    * otherwise              → keep previous position (starts at 0)

    Transaction cost ``config.LAMBDA_COST`` is charged on every position change.

    Returns
    -------
    Daily portfolio value ``pd.Series`` indexed by ``date``.
    """
    prices  = _load_prices(split)
    scores  = _load_scores(split, signal)
    merged  = prices.merge(scores, on=["ticker", "date"], how="left")
    merged  = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute persistent threshold positions per ticker
    records = []
    for ticker, grp in merged.groupby("ticker"):
        grp     = grp.reset_index(drop=True)
        pos     = np.zeros(len(grp), dtype=float)
        prev    = 0.0
        for i, row in grp.iterrows():
            s = row["score"]
            if pd.isna(s):
                pos[i] = prev
            elif s > threshold:
                pos[i] = 1.0
            elif s < -threshold:
                pos[i] = -1.0
            else:
                pos[i] = prev
            prev = pos[i]
        tmp = grp[["ticker", "date"]].copy()
        tmp["position"] = pos
        records.append(tmp)

    pos_df = pd.concat(records, ignore_index=True)
    return _portfolio_value(prices, pos_df, lambda_cost=config.LAMBDA_COST)


# ── B3 — Early fusion stub ────────────────────────────────────────────────────

def baseline_early_fusion(split: str = "val") -> None:
    """Placeholder — requires PPO training to be implemented first."""
    raise NotImplementedError(
        "Early fusion baseline requires PPO training — implement after T43"
    )


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(signal: str = "tech") -> float:
    """Grid-search the best threshold on the val split by Sharpe ratio.

    Returns
    -------
    Threshold that maximises annualised Sharpe on validation.
    """
    print(f"\nTuning threshold — signal='{signal}'")
    print(f"  {'threshold':>10}  {'Sharpe':>10}")
    print("  " + "-" * 24)

    best_thresh = _THRESH_GRID[0]
    best_sharpe = -np.inf

    for t in _THRESH_GRID:
        pv      = baseline_threshold(split="val", signal=signal, threshold=t)
        metrics = compute_metrics(pv)
        sharpe  = metrics["sharpe"]
        print(f"  {t:>10.2f}  {sharpe:>10.4f}")
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_thresh = t

    print(f"  → Best: threshold={best_thresh}  Sharpe={best_sharpe:.4f}")
    return best_thresh


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(portfolio_values: pd.Series) -> dict:
    """Compute standard performance metrics from a portfolio value series.

    Parameters
    ----------
    portfolio_values:
        Daily portfolio value indexed by date (first value need not be 1.0).

    Returns
    -------
    dict with keys: ``sharpe``, ``sortino``, ``max_drawdown``,
    ``cumulative_return``, ``calmar``.
    """
    pv   = portfolio_values.dropna()
    rets = pv.pct_change().dropna()

    mean_r = rets.mean()
    std_r  = rets.std(ddof=1)

    # Annualised Sharpe
    sharpe = float(np.sqrt(252) * mean_r / std_r) if std_r > 0 else 0.0

    # Sortino (downside deviation only)
    neg     = rets[rets < 0]
    std_neg = neg.std(ddof=1) if len(neg) > 1 else 1e-9
    sortino = float(np.sqrt(252) * mean_r / std_neg)

    # Maximum drawdown
    roll_max    = pv.cummax()
    drawdowns   = (pv - roll_max) / roll_max
    max_dd      = float(drawdowns.min())          # negative value

    # Cumulative return
    cum_ret = float(pv.iloc[-1] / pv.iloc[0] - 1)

    # Annualised return (CAGR)
    n_years    = len(rets) / 252
    ann_return = float((1 + cum_ret) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # Calmar
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "sharpe":            sharpe,
        "sortino":           sortino,
        "max_drawdown":      max_dd,
        "cumulative_return": cum_ret,
        "calmar":            calmar,
    }


# ── B4 — Early fusion PPO ─────────────────────────────────────────────────────

def train_early_fusion(out_name: str = "ppo_early_fusion") -> None:
    """Train a PPO agent with fusion_mode='early'.

    Uses the best hyperparameters from ``results/fusion/best_params.json`` and
    trains for 500 K timesteps.

    Parameters
    ----------
    out_name:
        Stem of the output model file (saved under ``results/fusion/``).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from src.environment.trading_env import TradingEnv
    from src.fusion.train import _N_EVAL_EPISODES, _ValSharpeCallback, evaluate_val

    fusion_mode = os.getenv("FUSION_MODE", "early")
    print(f"FUSION_MODE={fusion_mode!r}  (read from environment)")

    params_path = FUSION_DIR / "best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Run src/fusion/tune.py first — {params_path} not found.")

    params = json.loads(params_path.read_text())
    lambda_cost   = params["lambda_cost"]
    learning_rate = params["learning_rate"]
    ent_coef      = params["ent_coef"]
    gamma         = params["gamma"]

    print(f"Loaded best params: lr={learning_rate:.6f}  ent_coef={ent_coef:.4f}  "
          f"gamma={gamma:.4f}  lambda_cost={lambda_cost:.6f}\n")

    def _make_env():
        return TradingEnv(split="train", lambda_cost=lambda_cost, fusion_mode=fusion_mode)

    train_env = DummyVecEnv([_make_env] * 4)

    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = learning_rate,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        ent_coef      = ent_coef,
        gamma         = gamma,
        verbose       = 1,
    )

    FUSION_DIR.mkdir(parents=True, exist_ok=True)
    log_name = "training_log_early" if out_name == "ppo_early_fusion" else f"training_log_{out_name}"
    log_path = FUSION_DIR / f"{log_name}.csv"
    callback = _ValSharpeCallback(
        eval_freq       = 50_000,
        n_eval_episodes = _N_EVAL_EPISODES,
        log_path        = log_path,
        verbose         = 1,
    )

    print(f"Training PPO (fusion_mode='{fusion_mode}') for 500,000 timesteps …\n")
    model.learn(total_timesteps=500_000, callback=callback, progress_bar=True)
    train_env.close()

    final_sharpe = evaluate_val(model, n_episodes=_N_EVAL_EPISODES)
    print(f"\nFinal val Sharpe (early fusion, {_N_EVAL_EPISODES} episodes): {final_sharpe:+.4f}")

    out_path = FUSION_DIR / f"{out_name}.zip"
    model.save(str(out_path))
    print(f"Model saved → {out_path.relative_to(ROOT)}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true",
                    help="Save as ppo_early_fusion_v2.zip")
    args = ap.parse_args()

    sep = "=" * 50
    print(f"\n{sep}")
    print("  B4 — Early Fusion PPO")
    print(sep)
    out = "ppo_early_fusion_v2" if args.v2 else "ppo_early_fusion"
    train_early_fusion(out_name=out)
