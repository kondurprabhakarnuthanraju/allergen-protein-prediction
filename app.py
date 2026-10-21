"""
Protein Allergen Prediction — Streamlit app
Expected files:
  app.py
  models/allergen_cnn_bilstm.keras   (your existing trained model)
  models/metadata.json              (optional)
  requirements.txt

The app rebuilds the same CNN + BiLSTM architecture and loads weights from the
.keras archive if direct model deserialization fails. This avoids the
GlorotUniform(input_axes=...) config-deserialization error seen in some
Keras-version combinations.
"""

from pathlib import Path
import json
import tempfile
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import esm


# -------------------- Configuration --------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "allergen_cnn_bilstm.keras"
METADATA_PATH = MODEL_DIR / "metadata.json"
THRESHOLD = 0.57
EMBEDDING_SIZE = 320
MAX_SEQUENCE_LENGTH = 10000  # basic guard against accidental huge inputs


# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Protein Allergen Prediction",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Protein Allergen Prediction")
st.write(
    "Predict allergenicity from a protein amino-acid sequence using ESM-2 "
    "embeddings and a CNN–BiLSTM classifier."
)
st.caption(
    "Research/educational tool only. This prediction is not a clinical or "
    "food-safety determination."
)


# -------------------- Model architecture --------------------
def create_model():
    """Must match the architecture used when the model was trained."""
    model = keras.Sequential(
        [
            keras.Input(shape=(320, 1)),
            layers.Conv1D(128, kernel_size=5, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Conv1D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model_safely():
    """Try normal loading first; fall back to loading weights from .keras ZIP."""
    if not MODEL_PATH.exists():
        return None, (
            f"Trained model not found at `{MODEL_PATH}`. Add "
            "`models/allergen_cnn_bilstm.keras` to your GitHub repository."
        )

    try:
        loaded = keras.models.load_model(MODEL_PATH, compile=False)
        return loaded, None
    except Exception as direct_error:
        # A .keras file is a ZIP archive. Extract only its weights and load
