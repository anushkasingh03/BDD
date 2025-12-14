"""Simple CLI wrapper that can be reused outside scripts/run_game.py."""

from __future__ import annotations

from src.game.engine import GameEngine


def run_cli_game() -> None:
    engine = GameEngine.from_default_assets()
    engine.run_cli_session()
