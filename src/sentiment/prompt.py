"""Prompt templates and response parsing for LLM-based sentiment scoring."""
from __future__ import annotations

import json
import re

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a financial analyst. Your task is to assess the sentiment of a financial
news article with respect to the 5-day forward return of a specific stock.

You must respond ONLY with a valid JSON object in the following format:
{
  "sentiment": <float between -1.0 (very bearish) and 1.0 (very bullish)>,
  "confidence": <float between 0.0 (no confidence) and 1.0 (very confident)>,
  "reasoning": "<one sentence justifying the score>"
}

Do not include any text outside the JSON object.
"""

USER_TEMPLATE = """
Stock: {ticker}
Date: {date}
Headline: {headline}
Body: {body}

Assess the sentiment for {ticker} over the next 5 trading days.
"""

_BODY_MAX_CHARS = 500


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_llm_response(response_text: str) -> tuple[float, float]:
    """Parse a JSON response from the LLM and return (sentiment, confidence).

    Returns ``(0.0, 0.0)`` on any parse error or out-of-range value so that
    downstream code never receives invalid scores.

    Parameters
    ----------
    response_text:
        Raw string returned by the LLM.

    Returns
    -------
    (sentiment, confidence) both as floats, or (0.0, 0.0) on failure.
    """
    try:
        # Strip markdown code fences the model may emit despite instructions
        cleaned = re.sub(r"```(?:json)?|```", "", response_text).strip()
        data = json.loads(cleaned)

        sentiment  = float(data["sentiment"])
        confidence = float(data["confidence"])

        if not (-1.0 <= sentiment <= 1.0):
            return 0.0, 0.0
        if not (0.0 <= confidence <= 1.0):
            return 0.0, 0.0

        return sentiment, confidence

    except Exception:   # noqa: BLE001
        return 0.0, 0.0


# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_prompt(ticker: str, date: str, headline: str, body: str) -> str:
    """Fill ``USER_TEMPLATE`` with article metadata, truncating body.

    Parameters
    ----------
    ticker:
        Stock ticker symbol, e.g. ``"AAPL"``.
    date:
        Publication date string, e.g. ``"2024-03-15"``.
    headline:
        Article headline.
    body:
        Article body text.  Truncated to 500 characters.

    Returns
    -------
    Formatted user prompt string ready to send to the LLM.
    """
    return USER_TEMPLATE.format(
        ticker   = ticker,
        date     = date,
        headline = headline,
        body     = body[:_BODY_MAX_CHARS],
    )
