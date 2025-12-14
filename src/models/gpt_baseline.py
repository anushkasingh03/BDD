"""Pseudo GPT-based feedback baseline via OpenAI-compatible API stub."""

from __future__ import annotations

from typing import Iterable, List

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install openai to use the GPT baseline") from exc


class GPTFeedbackBaseline:
    """Lightweight helper that calls an LLM to label text as adaptive or distressed."""

    def __init__(self, model: str = "gpt-3.5-turbo", *, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def predict(self, texts: Iterable[str]) -> List[int]:  # pragma: no cover (network)
        outputs: List[int] = []
        for text in texts:
            prompt = (
                "Classify the following thought as ADAPTIVE (1) or DISTRESSED (0).\n"
                f"Thought: {text}\nReturn only 0 or 1."
            )
            resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            label = resp.choices[0].message.content.strip()
            outputs.append(int(label))
        return outputs
