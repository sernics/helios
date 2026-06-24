"""PPO late-fusion agent training.

Architecture
------------
State  : 7-dim vector from TradingEnv (sent + tech scores + 3 lagged returns)
Actions: Discrete(2) — hold / buy  (no short)
Reward : Sortino-based (see src/environment/reward.py)

Training uses 4 parallel TradingEnv(split='train') instances via DummyVecEnv.
Evaluation runs 10 full episodes on val every 50 K timesteps, logging the
mean annualised Sharpe of the resulting portfolio to a CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.environment.trading_env import TradingEnv

ROOT       = Path(__file__).resolve().parents[2]
RESULTS    = ROOT / "results" / "fusion"
MODEL_PATH = RESULTS / "ppo_late_fusion.zip"
LOG_PATH   = RESULTS / "training_log.csv"

_DEFAULT_PPO = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    ent_coef      = 0.01,
    gamma         = 0.99,
    verbose       = 1,
)
_TOTAL_TIMESTEPS = 500_000
_EVAL_FREQ       = 50_000
_N_EVAL_EPISODES = 10
_N_ENVS          = 4


# ── Val-Sharpe evaluation ──────────────────────────────────────────────────────

def _episode_sharpe(model: PPO, env: TradingEnv) -> float:
    """Run one complete episode deterministically; return annualised Sharpe."""
    obs, _ = env.reset()
    ep_df  = env._ep_df.copy()     # snapshot episode price data before any steps

    actions: list[int] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        actions.append(int(action))
        done = terminated or truncated

    # position[t] is entered at close of day t → earns log_ret of day t+1
    positions = np.array(actions[:-1], dtype=float)          # drop last (no t+1)
    log_rets  = ep_df["log_ret"].values[1 : len(actions)]     # aligned t+1 returns

    port_rets = positions * log_rets
    if len(port_rets) < 2 or port_rets.std() < 1e-9:
        return 0.0
    return float(np.sqrt(252) * port_rets.mean() / port_rets.std())


def evaluate_val(model: PPO, n_episodes: int = _N_EVAL_EPISODES) -> float:
    """Return mean Sharpe over *n_episodes* random val episodes."""
    env = TradingEnv(split="val", seed=0)
    sharpes = [_episode_sharpe(model, env) for _ in range(n_episodes)]
    return float(np.mean(sharpes))


# ── Custom callback ────────────────────────────────────────────────────────────

class _ValSharpeCallback(BaseCallback):
    """Evaluate on val every eval_freq timesteps and log to CSV."""

    def __init__(
        self,
        eval_freq: int,
        n_eval_episodes: int,
        log_path: Path,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path        = log_path
        self._next_eval      = eval_freq
        self._records: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            sharpe = evaluate_val(self.model, self.n_eval_episodes)
            self._records.append({"timestep": self.num_timesteps, "val_sharpe": sharpe})
            pd.DataFrame(self._records).to_csv(self.log_path, index=False)
            if self.verbose:
                print(
                    f"  [eval] timestep={self.num_timesteps:>7d}  "
                    f"val_sharpe={sharpe:+.4f}"
                )
            self._next_eval += self.eval_freq
        return True


# ── Training ───────────────────────────────────────────────────────────────────

def train_late_fusion(
    config_overrides: dict | None = None,
    out_name: str = "ppo_late_fusion",
    log_name: str = "training_log",
) -> PPO:
    """Train a PPO agent on the late-fusion trading environment.

    Parameters
    ----------
    config_overrides:
        Optional dict of PPO hyperparameter overrides (merged into defaults).
    out_name:
        Stem of the output model file (saved under ``results/fusion/``).
    log_name:
        Stem of the CSV training log file.

    Returns
    -------
    Trained ``stable_baselines3.PPO`` instance.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS / f"{out_name}.zip"
    log_path = RESULTS / f"{log_name}.csv"

    # Merge hyperparameters
    ppo_kwargs = {**_DEFAULT_PPO, **(config_overrides or {})}

    def _make_train_env():
        return TradingEnv(split="train")

    train_env = DummyVecEnv([_make_train_env] * _N_ENVS)

    verbose = ppo_kwargs.pop("verbose", 1)
    model   = PPO(env=train_env, verbose=verbose, **ppo_kwargs)

    callback = _ValSharpeCallback(
        eval_freq       = _EVAL_FREQ,
        n_eval_episodes = _N_EVAL_EPISODES,
        log_path        = log_path,
        verbose         = verbose,
    )

    print(f"\nTraining PPO for {_TOTAL_TIMESTEPS:,} timesteps "
          f"({_N_ENVS} parallel envs) …\n")

    model.learn(total_timesteps=_TOTAL_TIMESTEPS, callback=callback,
                progress_bar=True)

    final_sharpe = evaluate_val(model)
    print(f"\nFinal val Sharpe ({_N_EVAL_EPISODES} episodes): {final_sharpe:+.4f}")

    model.save(str(out_path))
    print(f"Model saved → {out_path.relative_to(ROOT)}")

    return model


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true",
                    help="Save as ppo_late_fusion_v2.zip")
    args = ap.parse_args()

    if args.v2:
        import json
        params_path = RESULTS / "best_params.json"
        params = json.loads(params_path.read_text())
        train_late_fusion(
            config_overrides={
                "learning_rate": params["learning_rate"],
                "ent_coef":      params["ent_coef"],
                "gamma":         params["gamma"],
            },
            out_name="ppo_late_fusion_v2",
            log_name="training_log_v2",
        )
    else:
        train_late_fusion()
