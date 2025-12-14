#!/usr/bin/env python3
"""CLI wrapper to train/evaluate models and save the pilot report."""

from __future__ import annotations

import argparse

from src.training.train_model import run_training_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BDD CBT sentiment model")
    parser.add_argument("--data", default="data/processed/train.csv", help="Path to processed CSV")
    parser.add_argument("--artifacts", default="artifacts", help="Output directory for reports/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training_job(data_path=args.data, artifacts_dir=args.artifacts)


if __name__ == "__main__":
    main()
