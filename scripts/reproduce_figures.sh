#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

from src.analysis import plot_distributions
from src.training.dataset import add_rule_sentiment, load_dataset

DATA = Path('data/processed/train.csv')
OUTPUT = Path('artifacts/plots')

data = add_rule_sentiment(load_dataset(DATA))
plot_distributions(data, OUTPUT)
print('Plots saved to', OUTPUT)
PY
