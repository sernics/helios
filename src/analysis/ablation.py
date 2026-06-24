"""Full ablation study on the test split (2024-2025).

Configurations
--------------
B1  buy-and-hold          always long, equal weight
B2  technical threshold   tau=0.20 on score_tech
B3  sentiment threshold   tau=0.20 on score_sent
B4  early-fusion PPO      ppo_early_fusion.zip
P   late-fusion PPO       ppo_late_fusion_tuned.zip

Regime analysis
---------------
Three market regimes are labelled using training-split statistics only:

  high_news  news_count > 75th pct of train news_count
  low_news   news_count ≤ 25th pct of train news_count  (i.e. = 0)
  high_vol   20-day realised portfolio vol > 1.5 × train mean vol

For the late-fusion (P) agent, mean conf_sent / conf_tech is reported per
regime to show how the model's input signal quality varies across conditions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.environment.trading_env import TradingEnv
from src.fusion.baselines import (
    _load_prices,
    _portfolio_value,
    baseline_buy_and_hold,
    baseline_threshold,
    compute_metrics,
    tune_threshold,
)

ROOT         = Path(__file__).resolve().parents[2]
SPLITS_DIR   = ROOT / "data" / "splits"
FUSION_DIR   = ROOT / "results" / "fusion"
ABLATION_DIR = ROOT / "results" / "ablation"

_TEST_SPLIT = "test"
_THRESHOLD  = 0.20  # fallback; per-signal thresholds are tuned on val


# ── RL agent runners ───────────────────────────────────────────────────────────

def _run_rl_agent(
    model: PPO,
    fusion_mode: str = "late",
    n_actions: int = 2,
) -> pd.Series:
    """Run a trained PPO agent on all test episodes; return portfolio value."""
    return _run_rl_agent_detailed(model, fusion_mode, n_actions)[0]


def _run_rl_agent_detailed(
    model: PPO,
    fusion_mode: str = "late",
    n_actions: int = 2,
) -> tuple[pd.Series, pd.DataFrame]:
    """Like _run_rl_agent but also returns per-step observation data.

    Returns
    -------
    pv : pd.Series
        Daily portfolio value indexed by date.
    step_df : pd.DataFrame
        One row per (ticker, date) with columns ``conf_sent``, ``conf_tech``
        (read from the observation vector before each action).

    Notes
    -----
    ``n_actions=2``  (v2 env): action 0=hold→pos 0, action 1=buy→pos +1
    ``n_actions=3``  (v1 env): action 0=sell→pos −1, 1=hold→0, 2=buy→+1
    """
    env    = TradingEnv(split=_TEST_SPLIT, fusion_mode=fusion_mode,
                        lambda_cost=config.LAMBDA_COST)
    prices = _load_prices(_TEST_SPLIT)

    pos_records:  list[dict] = []
    step_records: list[dict] = []

    for (ticker, year) in sorted(env._episodes.keys()):
        obs, _ = env.reset(options={"ticker": ticker, "year": year})
        ep_snap = env._ep_df.copy()

        step = 0
        done = False
        while not done:
            conf_sent = float(obs[1])
            conf_tech = float(obs[3])

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))

            date = ep_snap.iloc[step]["date"]
            # Position mapping depends on action-space size
            if n_actions == 2:
                position = float(int(action))        # 0→0, 1→+1
            else:
                position = float(int(action) - 1)    # 0→-1, 1→0, 2→+1
            pos_records.append({"ticker": ticker, "date": date,
                                 "position": position})
            step_records.append({"ticker": ticker, "date": date,
                                  "conf_sent": conf_sent, "conf_tech": conf_tech})
            step += 1
            done = terminated or truncated

    pos_df  = pd.DataFrame(pos_records)
    step_df = pd.DataFrame(step_records)
    pv      = _portfolio_value(prices, pos_df, lambda_cost=config.LAMBDA_COST)
    return pv, step_df


# ── Regime labelling ───────────────────────────────────────────────────────────

def _compute_regime_labels() -> tuple[pd.DataFrame, dict]:
    """Classify every (ticker, date) in the test split into market regimes.

    Thresholds are derived exclusively from the training split to avoid
    look-ahead bias.

    Returns
    -------
    regime_df : pd.DataFrame
        Columns: ticker, date, news_count, vol_20d,
                 is_high_news, is_low_news, is_high_vol.
    thresholds : dict
        The numeric cutoffs used (for reporting).
    """
    # ── Training thresholds ────────────────────────────────────────────────────
    train = pd.read_parquet(
        SPLITS_DIR / "train.parquet",
        columns=["ticker", "date", "close", "news_count"],
    )
    train["date"] = pd.to_datetime(train["date"])

    p25_news = float(train["news_count"].quantile(0.25))
    p75_news = float(train["news_count"].quantile(0.75))

    train["log_ret"] = train.groupby("ticker")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    train_port_vol = (
        train.groupby("date")["log_ret"].mean()
        .rolling(20).std().dropna().mean()
    )
    vol_threshold = float(train_port_vol * 1.5)

    # ── Test regime labels ─────────────────────────────────────────────────────
    test = pd.read_parquet(
        SPLITS_DIR / "test.parquet",
        columns=["ticker", "date", "close", "news_count"],
    )
    test["date"] = pd.to_datetime(test["date"])

    test["log_ret"] = test.groupby("ticker")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    test_port_vol_series = (
        test.groupby("date")["log_ret"].mean()
        .rolling(20).std()
        .rename("vol_20d")
        .reset_index()
    )

    test = test.merge(test_port_vol_series, on="date", how="left")
    test["is_high_news"] = test["news_count"] > p75_news
    test["is_low_news"]  = test["news_count"] <= p25_news
    test["is_high_vol"]  = test["vol_20d"] > vol_threshold

    thresholds = {
        "p25_news":      p25_news,
        "p75_news":      p75_news,
        "vol_threshold": vol_threshold,
    }
    return (
        test[["ticker", "date", "news_count", "vol_20d",
              "is_high_news", "is_low_news", "is_high_vol"]],
        thresholds,
    )


# ── Regime analysis ────────────────────────────────────────────────────────────

def run_regime_analysis(
    step_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    """Compute mean conf_sent / conf_tech per regime for the P agent.

    Parameters
    ----------
    step_df:
        Per-(ticker, date) observations from ``_run_rl_agent_detailed``.
    regime_df:
        Output of ``_compute_regime_labels``.
    thresholds:
        Numeric cutoffs (printed in the header).

    Returns
    -------
    DataFrame with regimes as rows and mean_conf_sent / mean_conf_tech as cols.
    """
    merged = step_df.merge(
        regime_df[["ticker", "date", "news_count", "vol_20d",
                   "is_high_news", "is_low_news", "is_high_vol"]],
        on=["ticker", "date"], how="left",
    )

    regime_specs = [
        ("all",       None),
        ("high_news", "is_high_news"),
        ("low_news",  "is_low_news"),
        ("high_vol",  "is_high_vol"),
        ("normal",    None),   # not high_vol AND not high_news AND not low_news
    ]

    rows = []
    for label, col in regime_specs:
        if label == "normal":
            sub = merged[
                ~merged["is_high_news"] & ~merged["is_low_news"] & ~merged["is_high_vol"]
            ]
        elif col is None:          # "all"
            sub = merged
        else:
            sub = merged[merged[col]]

        rows.append({
            "regime":         label,
            "n_obs":          len(sub),
            "n_dates":        sub["date"].nunique(),
            "mean_conf_sent": sub["conf_sent"].mean(),
            "mean_conf_tech": sub["conf_tech"].mean(),
            "mean_news_count": sub["news_count"].mean(),
        })

    result = pd.DataFrame(rows).set_index("regime")

    # ── Print ──────────────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  REGIME ANALYSIS — late-fusion P agent, test split 2024-2025")
    print(f"  Thresholds: news_count p25={thresholds['p25_news']:.0f}  "
          f"p75={thresholds['p75_news']:.0f}  "
          f"vol×1.5={thresholds['vol_threshold']:.5f}")
    print(sep)
    hdr = (f"  {'Regime':<12}  {'n_obs':>7}  {'n_dates':>7}  "
           f"{'mean_conf_sent':>14}  {'mean_conf_tech':>14}  {'mean_news':>9}")
    print(hdr)
    print("  " + "-" * 68)
    for regime, row in result.iterrows():
        print(
            f"  {regime:<12}  {int(row['n_obs']):>7d}  {int(row['n_dates']):>7d}  "
            f"{row['mean_conf_sent']:>14.4f}  {row['mean_conf_tech']:>14.4f}  "
            f"{row['mean_news_count']:>9.2f}"
        )
    print(sep)

    return result


# ── Main ablation ──────────────────────────────────────────────────────────────

def run_ablation() -> pd.DataFrame:
    """Evaluate all 5 configurations + regime analysis on the test split."""
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load RL models ────────────────────────────────────────────────────────
    print("Loading models …")
    late_model  = PPO.load(str(FUSION_DIR / "ppo_late_fusion_tuned.zip"))
    early_model = PPO.load(str(FUSION_DIR / "ppo_early_fusion.zip"))

    # ── Run all configurations ────────────────────────────────────────────────
    print("Running B1 — buy-and-hold …")
    b1_pv = baseline_buy_and_hold(split=_TEST_SPLIT)

    print("Tuning B2/B3 thresholds on val …")
    tau_tech = tune_threshold(signal="tech")
    tau_sent = tune_threshold(signal="sent")

    print(f"Running B2 — technical threshold (tau={tau_tech}) …")
    b2_pv = baseline_threshold(split=_TEST_SPLIT, signal="tech", threshold=tau_tech)

    print(f"Running B3 — sentiment threshold (tau={tau_sent}) …")
    b3_pv = baseline_threshold(split=_TEST_SPLIT, signal="sent", threshold=tau_sent)

    print("Running B4 — early-fusion PPO …")
    b4_pv = _run_rl_agent(early_model, fusion_mode="early")

    print("Running P  — late-fusion PPO (collecting observations for regime analysis) …")
    p_pv, p_step_df = _run_rl_agent_detailed(late_model, fusion_mode="late")

    portfolio_values = {
        "B1_BuyHold":     b1_pv,
        "B2_TechThresh":  b2_pv,
        "B3_SentThresh":  b3_pv,
        "B4_EarlyFusion": b4_pv,
        "P_LateFusion":   p_pv,
    }

    # ── Performance metrics ───────────────────────────────────────────────────
    rows = []
    for name, pv in portfolio_values.items():
        rows.append({"config": name, **compute_metrics(pv)})
    summary = pd.DataFrame(rows).set_index("config")

    sep = "=" * 78
    print(f"\n{sep}")
    print("  ABLATION STUDY — TEST SPLIT 2024-2025")
    print(sep)
    hdr = (f"  {'Config':<20}  {'Sharpe':>8}  {'Sortino':>8}  "
           f"{'MaxDD':>8}  {'CumRet':>8}  {'Calmar':>8}")
    print(hdr)
    print("  " + "-" * 74)
    for cfg, row in summary.iterrows():
        print(
            f"  {cfg:<20}  {row['sharpe']:>8.4f}  {row['sortino']:>8.4f}  "
            f"{row['max_drawdown']:>8.4f}  {row['cumulative_return']:>8.4f}  "
            f"{row['calmar']:>8.4f}"
        )
    print(sep)

    # ── Regime analysis ───────────────────────────────────────────────────────
    print("\nComputing regime labels …")
    regime_df, thresholds = _compute_regime_labels()
    regime_summary = run_regime_analysis(p_step_df, regime_df, thresholds)

    # ── Save all outputs ──────────────────────────────────────────────────────
    summary.to_csv(ABLATION_DIR / "summary.csv")
    regime_summary.to_csv(ABLATION_DIR / "regime_analysis.csv")

    equity = pd.DataFrame(portfolio_values).sort_index()
    equity.index.name = "date"
    equity.to_csv(ABLATION_DIR / "equity_curves.csv")

    print(f"\nSaved summary        → results/ablation/summary.csv")
    print(f"Saved regime analysis → results/ablation/regime_analysis.csv")
    print(f"Saved equity curves   → results/ablation/equity_curves.csv")

    return summary


# ── v2 Ablation (Sortino reward, 2-action env) ─────────────────────────────────

def run_ablation_v2() -> pd.DataFrame:
    """Evaluate all 5 configurations using v2 models (Discrete(2) + Sortino)."""
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading v2 models …")
    late_model  = PPO.load(str(FUSION_DIR / "ppo_late_fusion_v2.zip"))
    early_model = PPO.load(str(FUSION_DIR / "ppo_early_fusion_v2.zip"))

    print("Running B1 — buy-and-hold …")
    b1_pv = baseline_buy_and_hold(split=_TEST_SPLIT)

    print("Tuning B2/B3 thresholds on val …")
    tau_tech = tune_threshold(signal="tech")
    tau_sent = tune_threshold(signal="sent")

    print(f"Running B2 — technical threshold (tau={tau_tech}) …")
    b2_pv = baseline_threshold(split=_TEST_SPLIT, signal="tech", threshold=tau_tech)

    print(f"Running B3 — sentiment threshold (tau={tau_sent}) …")
    b3_pv = baseline_threshold(split=_TEST_SPLIT, signal="sent", threshold=tau_sent)

    print("Running B4v2 — early-fusion PPO v2 …")
    b4_pv = _run_rl_agent(early_model, fusion_mode="early", n_actions=2)

    print("Running Pv2  — late-fusion PPO v2 (collecting observations) …")
    p_pv, p_step_df = _run_rl_agent_detailed(late_model, fusion_mode="late", n_actions=2)

    portfolio_values = {
        "B1_BuyHold":      b1_pv,
        "B2_TechThresh":   b2_pv,
        "B3_SentThresh":   b3_pv,
        "B4v2_EarlyFusion": b4_pv,
        "Pv2_LateFusion":   p_pv,
    }

    rows = []
    for name, pv in portfolio_values.items():
        rows.append({"config": name, **compute_metrics(pv)})
    summary = pd.DataFrame(rows).set_index("config")

    sep = "=" * 78
    print(f"\n{sep}")
    print("  ABLATION STUDY v2 (Sortino + no-short) — TEST SPLIT 2024-2025")
    print(sep)
    hdr = (f"  {'Config':<22}  {'Sharpe':>8}  {'Sortino':>8}  "
           f"{'MaxDD':>8}  {'CumRet':>8}  {'Calmar':>8}")
    print(hdr)
    print("  " + "-" * 74)
    for cfg, row in summary.iterrows():
        print(
            f"  {cfg:<22}  {row['sharpe']:>8.4f}  {row['sortino']:>8.4f}  "
            f"{row['max_drawdown']:>8.4f}  {row['cumulative_return']:>8.4f}  "
            f"{row['calmar']:>8.4f}"
        )
    print(sep)

    # Regime analysis using v2 P agent
    print("\nComputing regime labels …")
    regime_df, thresholds = _compute_regime_labels()
    regime_summary = run_regime_analysis(p_step_df, regime_df, thresholds)

    summary.to_csv(ABLATION_DIR / "summary_v2.csv")
    regime_summary.to_csv(ABLATION_DIR / "regime_analysis_v2.csv")
    equity = pd.DataFrame(portfolio_values).sort_index()
    equity.index.name = "date"
    equity.to_csv(ABLATION_DIR / "equity_curves_v2.csv")

    print(f"\nSaved summary_v2        → results/ablation/summary_v2.csv")
    print(f"Saved regime_analysis_v2 → results/ablation/regime_analysis_v2.csv")
    print(f"Saved equity_curves_v2   → results/ablation/equity_curves_v2.csv")

    return summary


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true",
                    help="Run v2 ablation (ppo_*_v2.zip models)")
    args = ap.parse_args()

    if args.v2:
        run_ablation_v2()
    else:
        run_ablation()
