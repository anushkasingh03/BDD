"""Association tests (chi-square, correlation matrices)."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from scipy.stats import chi2_contingency


def chi_square_text_category(data: pd.DataFrame) -> Dict[str, float]:
    data = data.copy()
    data["text_category"] = pd.qcut(data["text_length"], q=2, labels=["Short", "Long"])
    contingency = pd.crosstab(data["text_category"], data["sentiment"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "contingency": contingency.values.tolist(),
        "expected": expected.tolist(),
    }
