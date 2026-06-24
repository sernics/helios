"""Gymnasium trading environment backed by real market + sentiment data.

State (7-dimensional Box in [-1, 1])
-------------------------------------
  idx  feature       normalisation
  0    score_sent    clipped to [-1, 1]   (already in range)
  1    conf_sent     kept as-is [0, 1] ⊆ [-1, 1]
  2    score_tech    clipped to [-1, 1]
  3    conf_tech     kept as-is [0, 1] ⊆ [-1, 1]
  4    ret_t-1       log-return clipped to [-0.1, 0.1], scaled by ×10 → [-1, 1]
  5    ret_t-2       same
  6    ret_t-3       same

Action space — Discrete(2): 0=hold, 1=buy  (no short selling)
Position p_t ∈ {0, 1}

Episode
-------
One episode = one (ticker, year) combination drawn from the loaded split.
``reset()`` samples a random episode; ``step()`` advances one trading day.
The episode ends after the last trading day of that year.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.environment.reward import sortino_reward

ROOT       = Path(__file__).resolve().parents[2]

load_dotenv(dotenv_path=ROOT / ".env.development", override=False)
load_dotenv(dotenv_path=ROOT / ".env", override=False)
SPLITS_DIR = ROOT / "data" / "splits"
TECH_DIR   = ROOT / "results" / "technical"

_RET_CLIP  = 0.1     # clip log-returns to ±10 %
_OBS_DIM   = 7


class TradingEnv(gym.Env):
    """Single-asset daily trading environment.

    Parameters
    ----------
    split:
        Which data split to use: ``"train"``, ``"val"``, or ``"test"``.
    lambda_cost:
        Transaction cost coefficient passed to :func:`sortino_reward`.
    seed:
        Optional RNG seed for episode sampling.
    fusion_mode:
        Override for FUSION_MODE env var (``"late"`` or ``"early"``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        split: str = "train",
        lambda_cost: float = 0.001,
        seed: int | None = None,
        fusion_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.split       = split
        self.lambda_cost = lambda_cost
        import os
        self.fusion_mode = fusion_mode or os.getenv("FUSION_MODE", "late")

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)   # 0=hold, 1=buy

        self._rng       = np.random.default_rng(seed)
        self._episodes  = self._build_episodes(split)
        self._ep_keys   = list(self._episodes.keys())

        # State set by reset()
        self._ep_df: pd.DataFrame | None = None
        self._step_idx    = 0
        self._prev_action = 0   # start with "hold"
        self._ticker      = ""
        self._year        = 0

    # ── Data loading ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_episodes(split: str) -> dict[tuple[str, int], pd.DataFrame]:
        """Return a dict mapping (ticker, year) → episode DataFrame."""
        # Load sentiment scores
        sent_df = pd.read_parquet(SPLITS_DIR / f"{split}_sentiment.parquet")
        sent_df["date"] = pd.to_datetime(sent_df["date"])

        # Load technical scores
        tech_df = pd.read_csv(
            TECH_DIR / f"{split}_scores.csv", parse_dates=["date"]
        )[["ticker", "date", "score_tech", "conf_tech"]]

        # Load raw split for close prices (needed for log-returns)
        price_df = pd.read_parquet(
            SPLITS_DIR / f"{split}.parquet", columns=["ticker", "date", "close"]
        )
        price_df["date"] = pd.to_datetime(price_df["date"])

        # Merge all features
        df = (
            sent_df
            .merge(tech_df, on=["ticker", "date"], how="inner")
            .merge(price_df, on=["ticker", "date"], how="inner")
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

        # Compute log-returns per ticker
        df["log_ret"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: np.log(s / s.shift(1)).fillna(0.0))
        )

        # Build (ticker, year) episode dict
        df["year"] = df["date"].dt.year
        episodes: dict[tuple[str, int], pd.DataFrame] = {}
        for (ticker, year), grp in df.groupby(["ticker", "year"]):
            episodes[(ticker, year)] = grp.reset_index(drop=True)

        return episodes

    # ── Observation builder ────────────────────────────────────────────────────

    def _make_obs(self, idx: int) -> np.ndarray:
        """Build the 7-dimensional normalised state vector for step *idx*."""
        row = self._ep_df.iloc[idx]

        def _ret(offset: int) -> float:
            i = idx - offset
            if i < 0:
                return 0.0
            r = float(self._ep_df.iloc[i]["log_ret"])
            return float(np.clip(r, -_RET_CLIP, _RET_CLIP) / _RET_CLIP)

        obs = np.array([
            float(np.clip(row["score_sent"], -1.0, 1.0)),
            float(np.clip(row["conf_sent"],   0.0, 1.0)),
            float(np.clip(row["score_tech"], -1.0, 1.0)),
            float(np.clip(row["conf_tech"],   0.0, 1.0)),
            _ret(1),
            _ret(2),
            _ret(3),
        ], dtype=np.float32)
        return obs

    # ── gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ticker = options.get("ticker") if options else None
        year   = options.get("year")   if options else None

        if ticker is not None and year is not None:
            key = (ticker, year)
            if key not in self._episodes:
                raise ValueError(f"Episode ({ticker}, {year}) not in {self.split} split.")
        else:
            key = self._ep_keys[int(self._rng.integers(len(self._ep_keys)))]

        self._ticker, self._year = key
        self._ep_df    = self._episodes[key]
        self._step_idx = 0
        self._prev_action = 0

        return self._make_obs(0), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._ep_df is not None, "Call reset() before step()."

        n = len(self._ep_df)

        # Gather the next-5 log-returns for the Sortino reward
        ret_start = self._step_idx + 1
        ret_end   = min(self._step_idx + 6, n)
        returns_window = self._ep_df["log_ret"].iloc[ret_start:ret_end].to_numpy()

        reward = sortino_reward(
            returns_window,
            action      = int(action),
            prev_action = self._prev_action,
            lambda_cost = self.lambda_cost,
        )

        row = self._ep_df.iloc[self._step_idx]
        info: dict[str, Any] = {
            "ticker":     self._ticker,
            "date":       str(row["date"].date()),
            "action":     int(action),
            "reward":     reward,
            "score_sent": float(row["score_sent"]),
            "score_tech": float(row["score_tech"]),
        }

        self._prev_action = int(action)
        self._step_idx   += 1
        terminated = self._step_idx >= n

        obs = (
            self._make_obs(self._step_idx)
            if not terminated
            else np.zeros(_OBS_DIM, dtype=np.float32)
        )

        return obs, reward, terminated, False, info
