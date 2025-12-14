"""High-level training orchestration for the dual-stream sentiment model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import tensorflow as tf

from src.analysis import chi_square_text_category, compute_text_length_tests, plot_distributions, run_regression
from src.models.dual_attention_model import DualStreamConfig, build_model
from src.training.dataset import add_rule_sentiment, load_dataset, stratified_train_test_split, tokenize_streams
from src.training.evaluate_model import evaluate_predictions
from src.utils import configure_logging, load_yaml, seed_everything

LOGGER = configure_logging(name=__name__)


def run_training_job(
    data_path: str | Path = "data/processed/train.csv",
    *,
    model_cfg_path: str | Path = "configs/model_config.yaml",
    training_cfg_path: str | Path = "configs/training_config.yaml",
    artifacts_dir: str | Path = "artifacts",
) -> Dict[str, object]:
    artifacts_dir = Path(artifacts_dir)
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = DualStreamConfig(**load_yaml(model_cfg_path))
    training_cfg = load_yaml(training_cfg_path)

    seed_everything(training_cfg.get("random_state", 42))

    data = load_dataset(data_path)
    data = add_rule_sentiment(data)

    tokenizer, context_seq, response_seq = tokenize_streams(
        data["Context"],
        data["Response"],
        vocab_size=model_cfg.vocab_size,
        seq_length=model_cfg.seq_length,
    )

    split = stratified_train_test_split(
        context_seq,
        response_seq,
        data["sentiment"],
        test_size=training_cfg["test_size"],
        random_state=training_cfg["random_state"],
    )

    model = build_model(model_cfg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        [split.X_train_context, split.X_train_response],
        split.y_train,
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
        validation_split=training_cfg["validation_split"],
        verbose=2,
    )

    y_pred_probs = model.predict([split.X_test_context, split.X_test_response], verbose=0)
    metrics = evaluate_predictions(split.y_test, y_pred_probs, report_dir=artifacts_dir)

    stats = compute_text_length_tests(data)
    regression = run_regression(data)
    chi_square = chi_square_text_category(data)
    plot_distributions(data, plots_dir)

    report = {
        "training_history": history.history,
        "metrics": metrics,
        "text_length_stats": stats,
        "regression_summary": regression.summary().as_text(),
        "chi_square": chi_square,
    }
    report_path = artifacts_dir / "pilot_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    tokenizer_path = artifacts_dir / "tokenizer.json"
    with open(tokenizer_path, "w", encoding="utf-8") as fp:
        json.dump(tokenizer.to_json(), fp)

    LOGGER.info("Training complete. Report saved to %s", report_path)
    return report
