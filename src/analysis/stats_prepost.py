"""Paired/group statistics for the pilot analyses."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from scipy.stats import f_oneway, ttest_ind


def compute_text_length_tests(data: pd.DataFrame) -> Dict[str, float]:
    group_neg = data[data["sentiment"] == 0]["text_length"]
    group_pos = data[data["sentiment"] == 1]["text_length"]

    t_stat, t_p = ttest_ind(group_neg, group_pos, equal_var=False)
    f_stat, f_p = f_oneway(group_neg, group_pos)

    pooled_std = (group_neg.std(ddof=1) ** 2 + group_pos.std(ddof=1) ** 2) ** 0.5
    effect_size = (group_neg.mean() - group_pos.mean()) / pooled_std

    return {
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "f_stat": float(f_stat),
        "f_p": float(f_p),
        "cohens_d": float(effect_size),
    }
