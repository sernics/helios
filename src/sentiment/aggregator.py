"""Daily sentiment aggregation across multiple news items.

Given a list of ``(sentiment_score, confidence)`` pairs for all news on a
single trading day / ticker combination, produces a single consolidated signal:

    s_sent = Σ(c_i * s_i) / Σ(c_i)   — confidence-weighted mean sentiment
    c_sent = mean(c_i)                 — average confidence

If the list is empty or all confidences are zero, both outputs are 0.0.
"""
from __future__ import annotations


def aggregate_daily(
    items: list[tuple[float, float]],
) -> tuple[float, float]:
    """Aggregate a day's news sentiment into a single (score, confidence) pair.

    Parameters
    ----------
    items:
        List of ``(sentiment_score, confidence)`` pairs, one per news article.
        ``sentiment_score`` should be in ``[-1, 1]`` and ``confidence`` in
        ``[0, 1]``.

    Returns
    -------
    (s_sent, c_sent)
        ``s_sent`` — confidence-weighted mean sentiment in ``[-1, 1]``.
        ``c_sent`` — mean confidence in ``[0, 1]``.
        Both are ``0.0`` when *items* is empty or all confidences are zero.
    """
    if not items:
        return 0.0, 0.0

    total_weight = sum(c for _, c in items)
    c_sent       = sum(c for _, c in items) / len(items)

    if total_weight == 0.0:
        return 0.0, 0.0

    s_sent = sum(c * s for s, c in items) / total_weight
    return s_sent, c_sent
