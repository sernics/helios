"""Optuna hyperparameter search for the PPO late-fusion agent.

Each trial trains PPO for 200 K timesteps, evaluates mean val Sharpe over 10
episodes, and returns it as the objective.  After *n_trials*, the best params
are used to retrain a final model for 500 K timesteps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.environment.trading_env import TradingEnv
from src.fusion.train import (
    LOG_PATH,
    MODEL_PATH,
    RESULTS,
    _N_ENVS,
    _N_EVAL_EPISODES,
    _ValSharpeCallback,
    evaluate_val,
)

ROOT             = Path(__file__).resolve().parents[2]
TUNED_MODEL_PATH = RESULTS / "ppo_late_fusion_tuned.zip"
BEST_PARAMS_PATH = RESULTS / "best_params.json"

_TRIAL_TIMESTEPS = 200_000
_FINAL_TIMESTEPS = 500_000

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_vec_env(lambda_cost: float, n_envs: int = _N_ENVS) -> DummyVecEnv:
    return DummyVecEnv([lambda: TradingEnv(split="train", lambda_cost=lambda_cost)] * n_envs)


# ── Objective ──────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    """Suggest PPO + env hyperparameters, train briefly, return val Sharpe."""
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    ent_coef      = trial.suggest_float("ent_coef",      0.0,  0.05)
    gamma         = trial.suggest_float("gamma",         0.95, 0.999)
    lambda_cost   = trial.suggest_float("lambda_cost",   5e-4, 5e-3, log=True)

    train_env = _make_vec_env(lambda_cost)
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = learning_rate,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        ent_coef      = ent_coef,
        gamma         = gamma,
        verbose       = 0,
    )
    model.learn(total_timesteps=_TRIAL_TIMESTEPS, progress_bar=False)
    train_env.close()

    sharpe = evaluate_val(model, n_episodes=_N_EVAL_EPISODES)
    trial.set_user_attr("val_sharpe", sharpe)
    return sharpe


# ── Study ──────────────────────────────────────────────────────────────────────

def run_tuning(n_trials: int = 20) -> dict:
    """Run Optuna study and retrain the final model with best params.

    Returns
    -------
    Best hyperparameters dict.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Running Optuna study — {n_trials} trials × {_TRIAL_TIMESTEPS:,} steps …")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best        = study.best_trial
    best_params = best.params
    best_sharpe = best.value

    print(f"\nBest trial #{best.number}  val_sharpe={best_sharpe:+.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k:<18} {v}")

    # ── Retrain final model on best params ─────────────────────────────────────
    lambda_cost = best_params["lambda_cost"]
    print(f"\nRetraining final model for {_FINAL_TIMESTEPS:,} timesteps …")

    final_env = _make_vec_env(lambda_cost)
    final_model = PPO(
        policy        = "MlpPolicy",
        env           = final_env,
        learning_rate = best_params["learning_rate"],
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        ent_coef      = best_params["ent_coef"],
        gamma         = best_params["gamma"],
        verbose       = 1,
    )

    tuned_log_path = RESULTS / "training_log_tuned.csv"
    callback = _ValSharpeCallback(
        eval_freq       = 50_000,
        n_eval_episodes = _N_EVAL_EPISODES,
        log_path        = tuned_log_path,
        verbose         = 1,
    )
    final_model.learn(
        total_timesteps = _FINAL_TIMESTEPS,
        callback        = callback,
        progress_bar    = True,
    )
    final_env.close()

    final_sharpe = evaluate_val(final_model, n_episodes=_N_EVAL_EPISODES)
    print(f"\nFinal val Sharpe (tuned model, {_N_EVAL_EPISODES} episodes): {final_sharpe:+.4f}")

    # ── Persist ────────────────────────────────────────────────────────────────
    final_model.save(str(TUNED_MODEL_PATH))

    params_to_save = {**best_params, "best_trial_sharpe": best_sharpe, "final_val_sharpe": final_sharpe}
    BEST_PARAMS_PATH.write_text(json.dumps(params_to_save, indent=2))

    print(f"\nSaved tuned model  → {TUNED_MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved best params  → {BEST_PARAMS_PATH.relative_to(ROOT)}")

    return best_params


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_tuning(n_trials=20)
