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
    """Rebuild the architecture recorded in the saved model's config.

    The saved classifier is NOT the earlier Sequential CNN with two LSTMs.
    Its actual structure is Conv1D(32, 3) -> BatchNorm -> ReLU -> Pool ->
    Dropout -> BiLSTM(32) -> Flatten -> Dense(64) -> Dropout -> Dense(2).
    Layer names and dimensions are kept aligned with the saved model so its
    trained weights can be loaded.
    """
    inputs = keras.Input(shape=(320, 1), name="embedding_input")
    x = layers.Conv1D(
        filters=32, kernel_size=3, padding="same", activation="linear",
        name="cnn_conv",
    )(inputs)
    x = layers.BatchNormalization(name="cnn_batch_norm")(x)
    x = layers.Activation("relu", name="cnn_relu")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same", name="cnn_max_pool")(x)
    x = layers.Dropout(0.15, name="cnn_dropout")(x)
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, name="forward_lstm"),
        name="bilstm",
    )(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.15, name="dense_dropout")(x)
    outputs = layers.Dense(2, activation="softmax", name="classifier")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="ESM2_CNN_BiLSTM_Allergen")


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
        # them into the architecture defined above, avoiding config parsing.
        try:
            with zipfile.ZipFile(MODEL_PATH, "r") as archive:
                weight_name = next(
                    (
                        name for name in archive.namelist()
                        if name.endswith("model.weights.h5")
                    ),
                    None,
                )
                if weight_name is None:
                    raise RuntimeError(
                        "No model.weights.h5 file was found inside the .keras archive."
                    )
                with tempfile.TemporaryDirectory() as temp_dir:
                    weight_path = Path(temp_dir) / "model.weights.h5"
                    weight_path.write_bytes(archive.read(weight_name))
                    rebuilt = create_model()
                    rebuilt.load_weights(weight_path)
                    return rebuilt, (
                        "Loaded the trained weights using the fallback loader "
                        "because direct Keras model loading was incompatible."
                    )
        except Exception as fallback_error:
            return None, (
                "Could not load the saved model.\n\n"
                f"Direct load error: {type(direct_error).__name__}: {direct_error}\n\n"
                f"Weight fallback error: {type(fallback_error).__name__}: "
                f"{fallback_error}\n\n"
                "Confirm the saved model was trained with the same CNN–BiLSTM "
                "architecture in this app."
            )


def get_class_names():
    """Read class order from metadata when present; otherwise use common label order."""
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            for key in ("class_names", "classes", "label_classes", "encoder_classes"):
                value = metadata.get(key)
                if isinstance(value, list) and len(value) == 2:
                    return [str(item) for item in value]
        except Exception:
            pass
    # LabelEncoder orders strings alphabetically. This fallback is only correct
    # if the training labels were the usual Allergenic/Non-Allergenic variants.
    return ["Allergenic", "Non-Allergenic"]


def get_allergen_probability(class_scores, class_names):
    """Return the score for whichever saved class label denotes allergenicity."""
    for i, name in enumerate(class_names):
        normalized = name.strip().lower().replace("_", "-").replace(" ", "-")
        if "allergen" in normalized and "non" not in normalized and "not" not in normalized:
            return float(class_scores[i])
    # If metadata labels are not recognizable, don't silently claim index 0 is allergen.
    return float(class_scores[0])


@st.cache_resource(show_spinner="Loading ESM-2 protein language model…")
def load_esm():
    """Load ESM-2 once per Streamlit server process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model.eval().to(device)
    return esm_model, alphabet, alphabet.get_batch_converter(), device


def clean_sequence(sequence):
    """Uppercase sequence and remove common FASTA formatting characters."""
    sequence = str(sequence).upper().strip()
    sequence = "".join(sequence.split())
    if sequence.startswith(">"):
        sequence = "".join(sequence.splitlines()[1:]).upper()
        sequence = "".join(sequence.split())
    # Gaps commonly appear in aligned sequences; remove them before embedding.
    sequence = sequence.replace("-", "").replace(".", "")
    return sequence


def validate_sequence(sequence):
    if not sequence:
        return "Enter a protein sequence."
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        return f"Sequence is too long (limit: {MAX_SEQUENCE_LENGTH:,} characters)."
    # Standard amino acids plus ambiguous/rare symbols accepted by ESM.
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        return "Unsupported character(s): " + ", ".join(invalid)
    if len(sequence) < 2:
        return "Sequence must contain at least two amino-acid characters."
    return None


def get_embedding(sequence, esm_model, batch_converter, device):
    """Return mean ESM-2 layer-6 residue embedding (320 features)."""
    _, _, tokens = batch_converter([("protein", sequence)])
    tokens = tokens.to(device)
    with torch.no_grad():
        results = esm_model(tokens, repr_layers=[6], return_contacts=False)
    token_representations = results["representations"][6]
    # Exclude beginning/end special tokens, then average residue embeddings.
    embedding = token_representations[0, 1:-1].mean(dim=0).cpu().numpy()
    if embedding.shape != (EMBEDDING_SIZE,):
        raise ValueError(
            f"Expected {EMBEDDING_SIZE} embedding values, got {embedding.shape}."
        )
    return embedding.astype(np.float32).reshape(1, EMBEDDING_SIZE, 1)


@st.cache_resource(show_spinner=False)
def get_classifier():
    return load_model_safely()


# -------------------- Load model --------------------
with st.sidebar:
    st.header("Model details")
    st.write("Embedding: ESM-2 t6 8M")
    st.write("Classifier: CNN + BiLSTM")
    st.write(f"Decision threshold: {THRESHOLD:.2f}")
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            with st.expander("Saved model metadata"):
                st.json(metadata)
        except Exception:
            st.caption("Metadata file exists but could not be read.")

classifier, model_message = get_classifier()
if classifier is None:
    st.error(model_message)
    st.info(
        "Expected repository layout: `app.py`, `requirements.txt`, and "
        "`models/allergen_cnn_bilstm.keras`."
    )
    st.stop()
elif model_message:
    st.warning(model_message)

# Load ESM only after confirming the classifier exists.
try:
    esm_model, alphabet, batch_converter, device = load_esm()
except Exception as exc:
    st.error(
        "Could not load the ESM-2 embedding model. Check that `fair-esm` and "
        f"`torch` are installed. Details: {type(exc).__name__}: {exc}"
    )
    st.stop()


# -------------------- Single-sequence prediction --------------------
single_tab, batch_tab, about_tab = st.tabs(
    ["Single sequence", "Batch prediction", "About"]
)

with single_tab:
    st.subheader("Enter a protein sequence")
    sequence_text = st.text_area(
        "Amino-acid sequence",
        height=180,
        placeholder="Example: MKWVTFISLLFLFSSAYS...",
        help="Paste the amino-acid sequence only. FASTA headers and line breaks are supported.",
    )

    if st.button("Predict allergenicity", type="primary"):
        sequence = clean_sequence(sequence_text)
        problem = validate_sequence(sequence)
        if problem:
            st.error(problem)
        else:
            try:
                with st.spinner("Generating ESM-2 embedding and predicting…"):
                    embedding = get_embedding(
                        sequence, esm_model, batch_converter, device
                    )
                    class_scores = classifier.predict(embedding, verbose=0)[0]
                class_index = int(np.argmax(class_scores))
                # The model has a 2-class softmax output, not a sigmoid output.
                # LabelEncoder sorts class labels alphabetically; use metadata if available.
                class_names = get_class_names()
                prediction = class_names[class_index]
                allergen_probability = get_allergen_probability(class_scores, class_names)
                st.subheader(f"Prediction: {prediction}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Allergen class score", f"{allergen_probability:.1%}")
                col2.metric("Predicted class score", f"{float(class_scores[class_index]):.1%}")
                col3.metric("Sequence length", f"{len(sequence):,} residues")
                st.progress(min(max(allergen_probability, 0.0), 1.0))
                st.caption(
                    "Scores are the model's two-class softmax outputs and are not "
                    "necessarily calibrated clinical probabilities. Class labels are "
                    "read from metadata when available."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")


# -------------------- Batch prediction --------------------
with batch_tab:
    st.subheader("Predict multiple sequences")
    st.write(
        "Upload a CSV or Excel file containing a column named `sequence`. "
        "The app adds prediction and score columns."
    )
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel dataset",
        type=["csv", "xlsx", "xls"],
        key="batch_upload",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            if "sequence" not in batch_df.columns:
                st.error(
                    "Your file must contain a column named `sequence`. "
                    f"Found columns: {', '.join(map(str, batch_df.columns))}"
                )
            else:
                st.write(f"Rows found: {len(batch_df):,}")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if len(batch_df) > 500:
                    st.warning(
                        "Large uploads can take a while. For reliability, process "
                        "500 sequences or fewer per upload."
                    )

                if st.button("Predict uploaded sequences", type="primary"):
                    results = []
                    progress = st.progress(0)
                    status = st.empty()
                    total = len(batch_df)

                    for index, raw_sequence in enumerate(batch_df["sequence"]):
                        sequence = clean_sequence(raw_sequence)
                        problem = validate_sequence(sequence)
                        if problem:
                            results.append(
                                {
                                    "cleaned_sequence": sequence,
                                    "sequence_length": len(sequence),
                                    "allergen_probability": np.nan,
                                    "prediction": "Invalid sequence",
                                    "error": problem,
                                }
                            )
                        else:
                            try:
                                embedding = get_embedding(
                                    sequence, esm_model, batch_converter, device
                                )
                                class_scores = classifier.predict(embedding, verbose=0)[0]
                                class_index = int(np.argmax(class_scores))
                                class_names = get_class_names()
                                allergen_probability = get_allergen_probability(class_scores, class_names)
                                results.append(
                                    {
                                        "cleaned_sequence": sequence,
                                        "sequence_length": len(sequence),
                                        "allergen_class_score": allergen_probability,
                                        "predicted_class_score": float(class_scores[class_index]),
                                        "prediction": class_names[class_index],
                                        "error": "",
                                    }
                                )
                            except Exception as exc:
                                results.append(
                                    {
                                        "cleaned_sequence": sequence,
                                        "sequence_length": len(sequence),
                                        "allergen_probability": np.nan,
                                        "prediction": "Error",
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                        progress.progress((index + 1) / max(total, 1))
                        status.write(f"Processed {index + 1} of {total}")

                    output_df = pd.concat(
                        [batch_df.reset_index(drop=True), pd.DataFrame(results)],
                        axis=1,
                    )
                    st.success("Batch prediction finished.")
                    st.dataframe(output_df, use_container_width=True)
                    st.download_button(
                        "Download predictions as CSV",
                        data=output_df.to_csv(index=False).encode("utf-8"),
                        file_name="allergen_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not read uploaded file: {type(exc).__name__}: {exc}")


with about_tab:
    st.subheader("How this app works")
    st.markdown(
        """
1. Cleans the input sequence by removing spaces, line breaks, and alignment gaps.
2. Generates a 320-dimensional mean embedding using ESM-2 layer 6.
3. Reshapes the embedding to `(1, 320, 1)`.
4. Runs the saved CNN–BiLSTM classifier, which produces two softmax class scores.
5. Selects the class with the highest score. Class order is read from metadata when available.
        """
    )
    st.warning(
        "Predictions depend on the dataset and training procedure used for the "
        "saved model. Do not use this app alone to make medical, clinical, or "
        "food-safety decisions."
    )
