"""FinBERT sentiment scorer using ProsusAI/finbert from HuggingFace.

The model outputs three-class probabilities: positive, negative, neutral.
We map them to a scalar signal:

    sentiment_score = p_positive - p_negative  →  [-1, 1]
    confidence      = max(p_positive, p_negative)  →  [0, 1]

Neutral probability is ignored for direction; a high neutral probability
naturally pushes both p_pos and p_neg down, which lowers confidence.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_NAME  = "ProsusAI/finbert"
_MAX_TOKENS  = 512


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FinBERTScorer:
    """Wraps ProsusAI/finbert for single-text and batch sentiment scoring."""

    def __init__(self, device: torch.device | None = None) -> None:
        self._device    = device or _best_device()
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model     = (
            AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
            .to(self._device)
        )
        self._model.eval()

        # FinBERT label order: positive=0, negative=1, neutral=2
        # Confirmed from the model card and config.json id2label.
        id2label = self._model.config.id2label
        self._pos_idx = next(i for i, l in id2label.items() if l.lower() == "positive")
        self._neg_idx = next(i for i, l in id2label.items() if l.lower() == "negative")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _forward(self, texts: list[str]) -> torch.Tensor:
        """Run a forward pass and return softmax probabilities (n, 3)."""
        inputs = self._tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = _MAX_TOKENS,
            return_tensors = "pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return torch.softmax(logits, dim=-1).cpu()

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, text: str) -> tuple[float, float]:
        """Score a single text string.

        Parameters
        ----------
        text:
            News headline or body text.

        Returns
        -------
        (sentiment_score, confidence)
            ``sentiment_score`` in ``[-1, 1]``, ``confidence`` in ``[0, 1]``.
        """
        probs = self._forward([text])[0]
        p_pos = probs[self._pos_idx].item()
        p_neg = probs[self._neg_idx].item()
        return p_pos - p_neg, max(p_pos, p_neg)

    def score_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        """Score a batch of texts.

        Parameters
        ----------
        texts:
            List of strings to score.

        Returns
        -------
        List of ``(sentiment_score, confidence)`` tuples in the same order as
        the input.
        """
        if not texts:
            return []
        probs = self._forward(texts)
        results = []
        for row in probs:
            p_pos = row[self._pos_idx].item()
            p_neg = row[self._neg_idx].item()
            results.append((p_pos - p_neg, max(p_pos, p_neg)))
        return results
