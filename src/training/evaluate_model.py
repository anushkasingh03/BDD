"""Model evaluation and reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_predictions(
    y_true,
    y_pred_probs,
    *,
    threshold: float = 0.5,
    report_dir: Path | None = None,
) -> Dict[str, float]:
    y_true = y_true.flatten()
    probs = y_pred_probs.flatten()
    y_pred = (probs > threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        np.save(report_dir / "predictions.npy", y_pred_probs)
    return metrics
