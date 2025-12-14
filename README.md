# BDD Gamified CBT Pilot Repo

A research-ready skeleton for studying a gamified CBT experience for body dysmorphic disorder/body-image distress. Users play through scenarios, enter free-text thoughts, the AI module analyzes sentiment + context, and CBT-aligned feedback routes back to the player.

## Repository layout
```
bdd-gamified-cbt/
├── configs/                # YAML configs for the model, training loop, and game
├── data/                   # raw + processed data placeholders (no PHI)
├── docs/                   # conceptual and API documentation
├── notebooks/              # analysis + modeling workflows
├── scripts/                # helper entry-points (game loop, training, figures)
├── src/                    # Python package (game, sentiment, models, training, analysis)
└── requirements.txt        # Python dependencies
```

## Getting started
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Place merged gameplay/questionnaire data at `data/processed/train.csv`.
4. Train + analyze: `python scripts/run_training.py` (or `bash scripts/run_training.sh`).
5. Playtest the CLI loop: `python scripts/run_game.py`.

Outputs land in `artifacts/` (created on demand): trained model weights, tokenizer, plots, `pilot_report.json`.

## Documentation
- `docs/method_overview.md` – Problem framing, study design, data streams, AI loop.
- `docs/CBT_mapping.md` – Scenario ↔ CBT technique mapping and sample scripts.
- `docs/API.md` – How to call model/game components from other services.
