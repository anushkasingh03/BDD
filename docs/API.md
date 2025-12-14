# API Overview

## Sentiment module
```python
from src.sentiment.polarity_features import extract_features
from src.models.dual_attention_model import build_model
```
- `extract_features(text)` returns lexicon-derived polarity scores, intensity, and length.
- `build_model(config)` constructs the dual-stream Keras model using `configs/model_config.yaml`.

## Training service
```python
from src.training.train_model import run_training_job
run_training_job(data_path="data/processed/train.csv")
```
Plans the tokenizer, trains models, logs metrics, saves outputs to `artifacts/`.

## Game engine
```python
from src.game.engine import GameEngine
engine = GameEngine.from_default_assets()
engine.run_cli_session()
```
Handles scenarios, routes sentiment outputs to CBT feedback, and records choices for future analysis.
