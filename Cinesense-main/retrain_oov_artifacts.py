"""Retrain IMDB sentiment model with OOV tokenizer; write paired artifacts.

Mirrors main.ipynb training. Deploy model.pkl and tokenizer.pkl together only.
Tokenizer config: num_words=5000, oov_token='<OOV>'
"""
from __future__ import annotations

import os
import pickle
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "imdb-dataset-of-50k-movie-reviews", "IMDB Dataset.csv")
ARTIFACT_DIR = os.path.join(ROOT, "cinesense", "review", "models")

NUM_WORDS = 5000
OOV_TOKEN = "<OOV>"
MAXLEN = 200
EPOCHS = 15
BATCH_SIZE = 64
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 2


def main() -> int:
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: dataset not found at {CSV_PATH}", file=sys.stderr)
        return 1

    print(f"Loading dataset: {CSV_PATH}")
    data = pd.read_csv(CSV_PATH)
    data["sentiment"] = (
        data["sentiment"].map({"positive": 1, "negative": 0}).astype("int32")
    )
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Tokenizer config: num_words={NUM_WORDS}, oov_token={OOV_TOKEN!r}")
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_data["review"])
    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_data["review"]), maxlen=MAXLEN
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(test_data["review"]), maxlen=MAXLEN
    )
    Y_train = train_data["sentiment"]
    Y_test = test_data["sentiment"]

    model = Sequential()
    model.add(Embedding(input_dim=NUM_WORDS, output_dim=128, input_length=MAXLEN))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.build(input_shape=(None, MAXLEN))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("Training with EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)...")
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
    )
    best_epoch = int(history.history["val_loss"].index(min(history.history["val_loss"]))) + 1
    print(
        f"Ran {len(history.history['val_loss'])} epoch(s); "
        f"best val_loss epoch={best_epoch} "
        f"val_loss={min(history.history['val_loss']):.4f}"
    )
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test loss: {loss} Test accuracy: {accuracy}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
    tokenizer_path = os.path.join(ARTIFACT_DIR, "tokenizer.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    # Notebook-local copies (same run)
    with open(os.path.join(ROOT, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(ROOT, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Saved paired artifacts:\n  {model_path}\n  {tokenizer_path}")
    print(f"oov_token={tokenizer.oov_token!r} index={tokenizer.word_index.get(OOV_TOKEN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
