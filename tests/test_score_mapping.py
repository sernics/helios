"""Unit tests for src/technical/score.py — predict_score signal mapping."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.technical.score import predict_score


# ── Minimal stub model ─────────────────────────────────────────────────────────

class _StubModel:
    """Minimal stand-in for XGBClassifier — accepts a pre-built proba array."""

    def __init__(self, probas: np.ndarray):
        # probas shape (n, 2): [[p_down, p_up], ...]
        self._probas = np.asarray(probas, dtype=float)

    def predict_proba(self, X):
        return self._probas


def _model(p_up_values):
    """Build a stub model from a list of p_up scalars."""
    p_up = np.asarray(p_up_values, dtype=float)
    return _StubModel(np.column_stack([1 - p_up, p_up]))


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_score_range():
    """score is always in [-1, 1]."""
    model  = _model(np.linspace(0, 1, 101))
    X_dummy = np.zeros((101, 1))
    score, _ = predict_score(model, X_dummy)
    assert np.all(score >= -1.0), "score below -1"
    assert np.all(score <= 1.0),  "score above +1"


def test_confidence_range():
    """confidence is always in [0, 1]."""
    model  = _model(np.linspace(0, 1, 101))
    X_dummy = np.zeros((101, 1))
    _, conf = predict_score(model, X_dummy)
    assert np.all(conf >= 0.0), "confidence below 0"
    assert np.all(conf <= 1.0), "confidence above 1"


def test_confidence_equals_abs_score():
    """confidence equals abs(score) exactly."""
    p_vals  = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    model   = _model(p_vals)
    X_dummy = np.zeros((len(p_vals), 1))
    score, conf = predict_score(model, X_dummy)
    np.testing.assert_array_equal(conf, np.abs(score))


def test_score_zero_at_half():
    """score is 0 when p_up = 0.5."""
    model   = _model([0.5])
    X_dummy = np.zeros((1, 1))
    score, _ = predict_score(model, X_dummy)
    assert score[0] == pytest.approx(0.0)


def test_score_one_at_full_bull():
    """score is +1 when p_up = 1.0."""
    model   = _model([1.0])
    X_dummy = np.zeros((1, 1))
    score, _ = predict_score(model, X_dummy)
    assert score[0] == pytest.approx(1.0)


def test_score_minus_one_at_full_bear():
    """score is -1 when p_up = 0.0."""
    model   = _model([0.0])
    X_dummy = np.zeros((1, 1))
    score, _ = predict_score(model, X_dummy)
    assert score[0] == pytest.approx(-1.0)
