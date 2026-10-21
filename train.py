import os, json, math, random, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, roc_auc_score, classification_report
)
import joblib
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Conv1D, Dropout, MaxPooling1D, Bidirectional, LSTM, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# def clean_sequence(seq):
#     seq = "".join(str(seq).split()).upper()
#     allowed = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
#     if not seq or any(ch not in allowed for ch in seq):
#         raise ValueError(f"Invalid protein sequence: {seq[:30]!r}")
#     return seq
def clean_sequence(seq):
    seq = "".join(str(seq).split()).upper()

    # Remove protein alignment gaps
    seq = seq.replace("-", "").replace(".", "")

    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZUO")

    if not seq or any(ch not in allowed for ch in seq):
        raise ValueError(f"Invalid protein sequence: {seq[:30]!r}")

    return seq

def find_dataset(explicit=None):
    candidates = [Path(explicit)] if explicit else []
    candidates += [
        ROOT / "pLM4Alg_protocol_example_allergens_short_dataset.xlsx",
        ROOT / "pLM4Alg_protocol_allergens_dataset.xlsx",
        ROOT / "pLM4Alg_protocol_allergens_dataset.csv",
        ROOT / "data" / "proteins.csv",
    ]
    for p in candidates:
        if p and p.exists():
            return p
    raise FileNotFoundError(
        "Dataset not found. Put the repository's labeled dataset in this folder or pass --data PATH. "
        "It must contain columns named 'sequence' and 'label'."
    )

def load_dataset(path):
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"sequence", "label"}.issubset(df.columns):
        raise ValueError(f"{path} must contain 'sequence' and 'label' columns; found {list(df.columns)}")
    df = df[["sequence", "label"]].dropna().copy()
    df["sequence"] = df["sequence"].map(clean_sequence)
    # Supports numeric 0/1 and common binary text labels, without guessing arbitrary labels.
    if not pd.api.types.is_numeric_dtype(df["label"]):
        mapping = {
            "0": 0, "non-allergen": 0, "non_allergen": 0, "non allergen": 0,
            "negative": 0, "1": 1, "allergen": 1, "positive": 1
        }
        labels = df["label"].astype(str).str.strip().str.lower().map(mapping)
        if labels.isna().any():
            raise ValueError("Text labels must be allergen/non-allergen or positive/negative.")
        df["label"] = labels
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both classes encoded as 0=non-allergen and 1=allergen.")
    # Exact duplicate sequences can cause leakage across train/test; keep one copy.
    df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    counts = df["label"].value_counts()
    if len(df) < 10 or counts.min() < 3:
        raise ValueError(f"Dataset too small for a reliable stratified split: {len(df)} rows; class counts={counts.to_dict()}")
    return df

def make_embeddings(sequences, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM-2 8M on {device} (first run downloads pretrained weights).")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model = esm_model.eval().to(device)
    converter = alphabet.get_batch_converter()
    vectors = []
    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start:start + batch_size]
        batch = [(f"seq_{start+i}", seq) for i, seq in enumerate(batch_seqs)]
        _, _, tokens = converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            reps = esm_model(tokens, repr_layers=[6], return_contacts=False)["representations"][6]
        for i, seq in enumerate(batch_seqs):
            # Exclude BOS/EOS; ESM-2 t6 has 320 features per residue.
            vec = reps[i, 1:len(seq)+1].mean(dim=0)
            vectors.append(vec.detach().cpu().numpy())
        del tokens, reps
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Embedded {min(start + batch_size, len(sequences))}/{len(sequences)}")
    return np.asarray(vectors, dtype=np.float32)

def build_model(filters=32, kernel_size=3, stride=1, dense_units=64, learning_rate=0.01):
    # Original protocol averages each protein to a 320-value vector. The CNN and BiLSTM
    # therefore operate over the 320 embedding dimensions, not over residue-level embeddings.
    inputs = Input(shape=(320, 1), name="embedding_input")
    x = Conv1D(filters, kernel_size, strides=stride, padding="same", name="cnn_conv")(inputs)
    x = BatchNormalization(name="cnn_batch_norm")(x)
    x = Activation("relu", name="cnn_relu")(x)
    x = MaxPooling1D(2, padding="same", name="cnn_max_pool")(x)
    x = Dropout(0.15, name="cnn_dropout")(x)
    x = Bidirectional(LSTM(32, return_sequences=True), name="bilstm")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(dense_units, activation="relu", name="fc1")(x)
    x = Dropout(0.15, name="dense_dropout")(x)
    outputs = Dense(2, activation="softmax", name="classifier")(x)
    model = Model(inputs, outputs, name="ESM2_CNN_BiLSTM_Allergen")
    model.compile(
        optimizer=SGD(learning_rate=learning_rate, momentum=0.5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def metrics_for(model, X, y):
    probs = model.predict(X, verbose=0)
    pred = np.argmax(probs, axis=1)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    try:
        auc = roc_auc_score(y, probs[:, 1])
    except ValueError:
        auc = float("nan")
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(auc),
        "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    print(classification_report(y, pred, labels=[0,1], target_names=["non-allergen", "allergen"], zero_division=0))
    return result

def main():
    parser = argparse.ArgumentParser(description="Train the ESM-2 + CNN + BiLSTM allergen classifier.")
    parser.add_argument("--data", default=None, help="CSV/XLSX with sequence,label columns")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reuse-embeddings", action="store_true", help="Reuse the repository's 320-d embedding CSV if row counts match.")
    args = parser.parse_args()

    data_path = find_dataset(args.data)
    df = load_dataset(data_path)
    print(f"Dataset: {data_path}; rows after deduplication: {len(df)}")
    print("Class counts (0=non-allergen, 1=allergen):", df.label.value_counts().sort_index().to_dict())

    emb_path = ROOT / "whole_sample_dataset_esm2_t6_8M_UR50D_unified_320_dimension.csv"
    X = None
    if args.reuse_embeddings and emb_path.exists():
        cached = pd.read_csv(emb_path)
        if cached.shape == (len(df), 320):
            X = cached.to_numpy(dtype=np.float32)
            print("Using cached 320-dimensional embeddings.")
        else:
            print(f"Cached embedding shape {cached.shape} does not match ({len(df)}, 320); regenerating.")
    if X is None:
        X = make_embeddings(df["sequence"].tolist())
        pd.DataFrame(X).to_csv(emb_path, index=False)
        print(f"Saved embeddings: {emb_path}")

    y = df["label"].to_numpy(dtype=np.int32)
    # Hold out test data before any fitting. Duplicate sequences were removed above.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    scaler = MinMaxScaler()
    X_fit = scaler.fit_transform(X_fit).reshape(-1, 320, 1)
    X_val = scaler.transform(X_val).reshape(-1, 320, 1)
    X_test_scaled = scaler.transform(X_test).reshape(-1, 320, 1)

    model = build_model()
    model.summary()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ]
    model.fit(
        X_fit, y_fit, validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks,
        verbose=1
    )
    test_metrics = metrics_for(model, X_test_scaled, y_test)
    print("Held-out test metrics:")
    print(json.dumps(test_metrics, indent=2))

    model.save(MODELS / "allergen_cnn_bilstm.keras")
    joblib.dump(scaler, MODELS / "scaler.joblib")
    (MODELS / "metadata.json").write_text(json.dumps({
        "embedding_model": "esm2_t6_8M_UR50D",
        "embedding_dimension": 320,
        "architecture": "Conv1D -> BatchNorm -> ReLU -> MaxPool -> Dropout -> BiLSTM -> Flatten -> Dense -> Softmax",
        "label_mapping": {"0": "non-allergen", "1": "allergen"},
        "dataset_file": str(data_path.name),
        "n_samples_after_deduplication": int(len(df)),
        "test_metrics": test_metrics,
        "warning": "Metrics depend on dataset quality and split; similarity/homology leakage has not been ruled out."
    }, indent=2), encoding="utf-8")
    print(f"Saved model and scaler in {MODELS}")
    print("Training complete.")

if __name__ == "__main__":
    main()
