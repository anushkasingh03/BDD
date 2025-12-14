"""Plotting helpers for paper-quality figures."""

from __future__ import annotations

from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


def plot_distributions(data: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.boxplot(x="sentiment", y="text_length", data=data)
    plt.title("Text Length Distribution Across Sentiments")
    plt.savefig(output_dir / "text_length_boxplot.png", bbox_inches="tight")
    plt.close()

    sns.violinplot(x="sentiment", y="text_length", data=data)
    plt.title("Text Length Variability Across Sentiments")
    plt.savefig(output_dir / "text_length_violin.png", bbox_inches="tight")
    plt.close()
