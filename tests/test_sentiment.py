"""Unit tests for the sentiment module.

Tests 1-6  — pure-Python logic (prompt + aggregator), no model needed.
Tests 7-8  — FinBERT inference (downloads model weights on first run).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sentiment.aggregator import aggregate_daily
from src.sentiment.prompt import parse_llm_response

# ── parse_llm_response ─────────────────────────────────────────────────────────

def test_parse_invalid_json_returns_zero():
    """Returns (0.0, 0.0) on invalid JSON."""
    s, c = parse_llm_response("not json at all")
    assert (s, c) == (0.0, 0.0)


def test_parse_sentiment_out_of_range_returns_zero():
    """Returns (0.0, 0.0) when sentiment is outside [-1, 1]."""
    payload = '{"sentiment": 2.5, "confidence": 0.8, "reasoning": "too high"}'
    s, c = parse_llm_response(payload)
    assert (s, c) == (0.0, 0.0)


def test_parse_valid_response():
    """Correctly parses a well-formed LLM response."""
    payload = '{"sentiment": 0.75, "confidence": 0.9, "reasoning": "strong earnings beat"}'
    s, c = parse_llm_response(payload)
    assert s == pytest.approx(0.75)
    assert c == pytest.approx(0.9)


def test_parse_strips_markdown_fences():
    """Strips ```json ... ``` fences before parsing."""
    payload = '```json\n{"sentiment": -0.5, "confidence": 0.6, "reasoning": "miss"}\n```'
    s, c = parse_llm_response(payload)
    assert s == pytest.approx(-0.5)
    assert c == pytest.approx(0.6)


# ── aggregate_daily ────────────────────────────────────────────────────────────

def test_aggregate_empty_list():
    """Returns (0.0, 0.0) when the input list is empty."""
    assert aggregate_daily([]) == (0.0, 0.0)


def test_aggregate_confidence_weighted_mean():
    """Computes confidence-weighted mean sentiment and mean confidence."""
    items = [
        (0.8, 0.9),   # high-confidence bullish
        (-0.4, 0.1),  # low-confidence bearish
    ]
    s, c = aggregate_daily(items)
    # s = (0.9*0.8 + 0.1*(-0.4)) / (0.9+0.1) = (0.72 - 0.04) / 1.0 = 0.68
    assert s == pytest.approx(0.68, abs=1e-6)
    # c = mean(0.9, 0.1) = 0.5
    assert c == pytest.approx(0.5, abs=1e-6)


def test_aggregate_all_zero_confidences():
    """Returns (0.0, 0.0) when all confidences are zero."""
    items = [(0.5, 0.0), (-0.3, 0.0)]
    assert aggregate_daily(items) == (0.0, 0.0)


# ── FinBERT ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def finbert():
    from src.sentiment.finbert import FinBERTScorer
    return FinBERTScorer()


def test_finbert_score_ranges(finbert):
    """score() returns sentiment in [-1, 1] and confidence in [0, 1]."""
    s, c = finbert.score("Apple reported record quarterly earnings, beating expectations.")
    assert -1.0 <= s <= 1.0, f"sentiment {s} out of range"
    assert  0.0 <= c <= 1.0, f"confidence {c} out of range"


def test_finbert_score_batch_length(finbert):
    """score_batch() returns the same number of results as inputs."""
    texts = [
        "Company beats earnings estimates for the third consecutive quarter.",
        "Shares plunge after CEO announces surprise resignation.",
        "Markets remain flat amid mixed economic data.",
    ]
    results = finbert.score_batch(texts)
    assert len(results) == len(texts)
    for s, c in results:
        assert -1.0 <= s <= 1.0
        assert  0.0 <= c <= 1.0
