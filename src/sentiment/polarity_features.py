"""Feature extraction for sentiment routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from textblob import TextBlob


def _safe_length(text: str) -> int:
    return len(text.split()) if text else 0


def extract_features(text: str) -> Dict[str, float]:
    blob = TextBlob(text or "")
    sentiment = blob.sentiment
    tokens = blob.words
    return {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity,
        "token_count": float(len(tokens)),
        "avg_token_length": (len(text) / max(len(tokens), 1)) if tokens else 0.0,
    }
