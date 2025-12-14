"""Dataset utilities for training/evaluating sentiment models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from textblob import TextBlob


@dataclass
class DatasetSplit:
    X_train_context: np.ndarray
    X_test_context: np.ndarray
    X_train_response: np.ndarray
    X_test_response: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"Context", "Response"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset missing columns: {required - set(df.columns)}")
    df["Context"] = df["Context"].astype(str)
    df["Response"] = df["Response"].astype(str)
    return df


def add_rule_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = df["Response"].apply(_blob_label)
    df["text_length"] = df["Context"].str.len()
    return df


def _blob_label(text: str) -> int:
    blob = TextBlob(text or "")
    return int(blob.sentiment.polarity > 0)


def tokenize_streams(
    contexts: pd.Series,
    responses: pd.Series,
    *,
    vocab_size: int,
    seq_length: int,
) -> Tuple[Tokenizer, np.ndarray, np.ndarray]:
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(contexts + " " + responses)
    ctx_seq = tokenizer.texts_to_sequences(contexts)
    rsp_seq = tokenizer.texts_to_sequences(responses)
    ctx_seq = pad_sequences(ctx_seq, maxlen=seq_length)
    rsp_seq = pad_sequences(rsp_seq, maxlen=seq_length)
    return tokenizer, ctx_seq, rsp_seq


def stratified_train_test_split(
    context_seq: np.ndarray,
    response_seq: np.ndarray,
    labels: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    (
        X_train_context,
        X_test_context,
        X_train_response,
        X_test_response,
        y_train,
        y_test,
    ) = train_test_split(
        context_seq,
        response_seq,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return DatasetSplit(
        X_train_context=X_train_context,
        X_test_context=X_test_context,
        X_train_response=X_train_response,
        X_test_response=X_test_response,
        y_train=y_train,
        y_test=y_test,
    )
