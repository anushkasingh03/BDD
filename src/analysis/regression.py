"""Regression helpers for predicting symptom change."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def run_regression(data: pd.DataFrame):
    predictors = ["text_length"]
    valid = data.dropna(subset=predictors + ["sentiment"])
    design = sm.add_constant(valid[predictors])
    model = sm.OLS(valid["sentiment"], design).fit()
    return model
