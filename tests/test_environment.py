"""Tests for src/environment/trading_env.py and reward.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.reward import sortino_reward
from src.environment.trading_env import TradingEnv


# ── Shared fixture (env loaded once per session) ───────────────────────────────

@pytest.fixture(scope="session")
def env():
    return TradingEnv(split="train", lambda_cost=0.001, seed=0)


# ── 1. 100 random episodes complete without errors ─────────────────────────────

def test_100_episodes_complete(env):
    """100 randomly sampled episodes run to termination without exception."""
    for i in range(100):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated


# ── 2. Reward is always finite ─────────────────────────────────────────────────

def test_reward_finite(env):
    """All rewards during 20 random episodes are finite (no NaN or inf)."""
    for _ in range(20):
        env.reset()
        done = False
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            done = terminated or truncated


# ── 3. State values are all in [-1, 1] ────────────────────────────────────────

def test_observation_bounds(env):
    """Every observation component stays within [-1, 1] across 20 episodes."""
    for _ in range(20):
        obs, _ = env.reset()
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0), \
            f"Initial obs out of range: {obs}"
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            if not done:
                assert np.all(obs >= -1.0) and np.all(obs <= 1.0), \
                    f"Obs out of range: {obs}"


# ── 4. Action space is Discrete(2) ────────────────────────────────────────────

def test_action_space_size(env):
    """Action space must be Discrete(2): hold=0, buy=1."""
    assert env.action_space.n == 2, \
        f"Expected Discrete(2), got Discrete({env.action_space.n})"


# ── 5. Hold action never incurs transaction cost ───────────────────────────────

def test_hold_no_cost():
    """hold → hold incurs zero transaction cost."""
    rw = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    r_hold  = sortino_reward(rw, action=0, prev_action=0, lambda_cost=0.001)
    r_no_lc = sortino_reward(rw, action=0, prev_action=0, lambda_cost=0.0)
    assert r_hold == pytest.approx(r_no_lc), \
        "hold → hold should not incur any cost"


# ── 6. Position change incurs transaction cost ────────────────────────────────

def test_position_change_incurs_cost():
    """Switching from hold to buy (or buy to hold) deducts lambda_cost."""
    rw = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    lc = 0.001

    # hold → buy
    r_switch  = sortino_reward(rw, action=1, prev_action=0, lambda_cost=lc)
    r_no_cost = sortino_reward(rw, action=1, prev_action=0, lambda_cost=0.0)
    assert r_switch == pytest.approx(r_no_cost - lc, abs=1e-9), \
        "hold → buy should deduct lambda_cost exactly once"

    # buy → hold
    r_switch2 = sortino_reward(rw, action=0, prev_action=1, lambda_cost=lc)
    r_no_cost2 = sortino_reward(rw, action=0, prev_action=1, lambda_cost=0.0)
    assert r_switch2 == pytest.approx(r_no_cost2 - lc, abs=1e-9), \
        "buy → hold should deduct lambda_cost exactly once"


# ── 7. Episode length matches trading days in that ticker-year ────────────────

def test_episode_length(env):
    """Episode step count equals the number of rows for that (ticker, year)."""
    for (ticker, year), ep_df in list(env._episodes.items())[:10]:
        obs, _ = env.reset(options={"ticker": ticker, "year": year})
        expected_len = len(ep_df)
        steps = 0
        done  = False
        while not done:
            _, _, terminated, truncated, _ = env.step(0)   # always hold
            done = terminated or truncated
            steps += 1
        assert steps == expected_len, \
            f"({ticker}, {year}): expected {expected_len} steps, got {steps}"
