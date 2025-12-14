"""Dual-stream CNN + BiLSTM with custom attention for the CBT router."""

from __future__ import annotations

from dataclasses import dataclass

from tensorflow.keras import Model, layers


@dataclass(frozen=True)
class DualStreamConfig:
    vocab_size: int = 10_000
    seq_length: int = 100
    embedding_dim: int = 128
    lstm_units: int = 64
    conv_filters: int = 64
    conv_kernel_size: int = 3
    dense_units: int = 64
    memory_units: int = 128
    dropout_rate: float = 0.5


def build_model(config: DualStreamConfig = DualStreamConfig()) -> Model:
    context_input = layers.Input(shape=(config.seq_length,), name="Context_Input")
    response_input = layers.Input(shape=(config.seq_length,), name="Response_Input")

    embedding_layer = layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embedding_dim,
        name="shared_embedding",
    )
    context_embed = embedding_layer(context_input)
    response_embed = embedding_layer(response_input)

    conv_kwargs = dict(filters=config.conv_filters, kernel_size=config.conv_kernel_size, activation="relu", padding="same")
    context_conv = layers.Conv1D(**conv_kwargs)(context_embed)
    response_conv = layers.Conv1D(**conv_kwargs)(response_embed)

    bi_lstm = lambda x: layers.Bidirectional(layers.LSTM(config.lstm_units, return_sequences=True))(x)
    context_bi = bi_lstm(context_conv)
    response_bi = bi_lstm(response_conv)

    concat = layers.Concatenate()
    context_cross = concat([context_bi, response_bi])
    response_cross = concat([response_bi, context_bi])

    attention = layers.Dense(256, activation="tanh")
    context_att_weights = layers.Activation("softmax")(attention(context_cross))
    response_att_weights = layers.Activation("softmax")(attention(response_cross))

    context_vec = layers.Dot(axes=1)([context_att_weights, context_cross])
    response_vec = layers.Dot(axes=1)([response_att_weights, response_cross])

    memory = layers.Add()([context_vec, response_vec])
    memory = layers.Dense(config.memory_units, activation="relu")(memory)

    final = layers.Concatenate()([layers.Flatten()(context_vec), layers.Flatten()(response_vec), layers.Flatten()(memory)])
    final = layers.Dense(config.dense_units, activation="relu")(final)
    final = layers.Dropout(config.dropout_rate)(final)
    output = layers.Dense(1, activation="sigmoid", name="sentiment_output")(final)

    model = Model(inputs=[context_input, response_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
