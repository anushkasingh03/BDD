"""Simple sentiment baselines for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def _lexicon_score(text: str, lexicon: Dict[str, float]) -> float:
    score = 0.0
    tokens = text.lower().split()
    for token in tokens:
        score += lexicon.get(token, 0.0)
    return score / max(len(tokens), 1)


@dataclass
class LexiconRuleBaseline:
    lexicon: Dict[str, float]
    threshold: float = 0.0

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return np.array([float(_lexicon_score(t, self.lexicon) > self.threshold) for t in texts])


def train_logistic_baseline(texts: Iterable[str], labels: Iterable[int]) -> Pipeline:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(list(texts), list(labels))
    return pipeline
