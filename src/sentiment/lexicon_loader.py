"""Load sentiment lexicons for the rule-based baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def load_lexicon(path: str | Path) -> Dict[str, float]:
    """Simple TSV lexicon loader (token\tscore per line)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")
    lexicon: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token, score = line.split("\t")
            lexicon[token.lower()] = float(score)
    return lexicon
