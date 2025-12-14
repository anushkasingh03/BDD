"""Minimal serious-game loop for the BDD CBT prototype."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.sentiment.polarity_features import extract_features
from src.utils.logging_utils import configure_logging

LOGGER = configure_logging(name=__name__)


@dataclass
class Choice:
    id: str
    text: str
    health: int


@dataclass
class Scenario:
    id: str
    title: str
    prompt: str
    cbt_tags: List[str]
    choices: List[Choice]


@dataclass
class Feedback:
    label: str
    message: str
    skills: List[str]


class GameEngine:
    def __init__(self, scenarios: List[Scenario], feedback_map: Dict[str, Feedback]):
        self.scenarios = scenarios
        self.feedback_map = feedback_map
        self.session_log: List[Dict[str, str]] = []

    @classmethod
    def from_default_assets(
        cls,
        scenario_path: Path | str = Path("src/game/scenarios.json"),
        feedback_path: Path | str = Path("src/game/feedback_templates.json"),
    ) -> "GameEngine":
        scenarios = _load_scenarios(scenario_path)
        feedback = _load_feedback(feedback_path)
        return cls(scenarios, feedback)

    def _route_feedback(self, text: str) -> Feedback:
        features = extract_features(text)
        polarity = features["polarity"]
        if polarity <= -0.3:
            zone = "strongly_negative"
        elif polarity < 0:
            zone = "mildly_negative"
        elif polarity < 0.3:
            zone = "neutral"
        else:
            zone = "positive"
        LOGGER.debug("Routing text '%s' with polarity %.3f to zone %s", text, polarity, zone)
        return self.feedback_map[zone]

    def run_cli_session(self) -> None:
        LOGGER.info("Starting CLI CBT session")
        for scenario in self.scenarios:
            print(f"\n== {scenario.title} ==")
            print(scenario.prompt)
            text = input("Your thought: ")
            feedback = self._route_feedback(text)
            print(f"\n{feedback.label}: {feedback.message}")
            print("Suggested skills:", ", ".join(feedback.skills))

            for idx, choice in enumerate(scenario.choices, start=1):
                print(f"  [{idx}] {choice.text}")
            selection = input("Pick an action #: ")
            try:
                choice = scenario.choices[int(selection) - 1]
            except Exception:  # pragma: no cover - interactive
                LOGGER.warning("Invalid choice; defaulting to option 1")
                choice = scenario.choices[0]

            log_entry = {
                "scenario_id": scenario.id,
                "user_text": text,
                "feedback_label": feedback.label,
                "choice_id": choice.id,
                "health_score": choice.health,
            }
            self.session_log.append(log_entry)
            LOGGER.info("Scenario %s logged", scenario.id)
        print("\nSession complete! Review session_log for analytics.")


def _load_scenarios(path: Path | str) -> List[Scenario]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios: List[Scenario] = []
    for item in data:
        choices = [Choice(**choice) for choice in item["choices"]]
        scenarios.append(
            Scenario(
                id=item["id"],
                title=item["title"],
                prompt=item["prompt"],
                cbt_tags=item.get("cbt_tags", []),
                choices=choices,
            )
        )
    return scenarios


def _load_feedback(path: Path | str) -> Dict[str, Feedback]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {zone: Feedback(**payload) for zone, payload in data.items()}
