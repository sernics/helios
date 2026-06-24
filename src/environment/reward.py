"""Reward function for the trading environment.

The reward blends a 5-day forward Sortino ratio with a transaction-cost
penalty each time the agent changes its position.
"""
from __future__ import annotations

import numpy as np

_ANNUALISE  = np.sqrt(252)
_CLIP_LIMIT = 3.0


def sortino_reward(
    returns_window: np.ndarray,
    action: int,
    prev_action: int,
    lambda_cost: float = 0.001,
) -> float:
    """Compute the single-step reward using a Sortino-based signal.

    Parameters
    ----------
    returns_window:
        Array of the next 5 daily log-returns ``r_{t+1} … r_{t+5}``.
        If fewer than 5 returns are available, returns ``0.0``.
    action:
        Current action (0=hold, 1=buy).
    prev_action:
        Previous action.  Transaction cost is charged when this differs
        from *action*.
    lambda_cost:
        Cost coefficient per trade (default 0.001).

    Returns
    -------
    Scalar reward (float).
    """
    if len(returns_window) < 5:
        return 0.0

    r        = np.asarray(returns_window, dtype=float)
    mean_r   = np.mean(r)
    downside = r[r < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-8
    sortino  = float(np.clip(_ANNUALISE * mean_r / (downside_std + 1e-8),
                             -_CLIP_LIMIT, _CLIP_LIMIT))
    cost = lambda_cost if action != prev_action else 0.0
    return sortino - cost
