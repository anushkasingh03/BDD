"""Wrapper for using a HuggingFace BERT sentiment checkpoint as baseline."""

from __future__ import annotations

from typing import Iterable, List

try:
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install transformers to use the BERT baseline") from exc


class BertSentimentBaseline:
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        self._pipeline = pipeline("sentiment-analysis", model=model_name)

    def predict(self, texts: Iterable[str]) -> List[float]:
        outputs = self._pipeline(list(texts))
        # Map label to pseudo binary score (>=3 stars => adaptive)
        scores = []
        for result in outputs:
            label = result["label"]
            score = 1.0 if label.endswith(("4", "5")) else 0.0
            scores.append(score)
        return scores
