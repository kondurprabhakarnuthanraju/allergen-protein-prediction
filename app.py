import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import esm
import joblib
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "allergen_cnn_bilstm.keras"
SCALER_PATH = ROOT / "models" / "scaler.joblib"
META_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Protein Allergen Prediction", page_icon="🧬", layout="centered")
st.title("🧬 Protein Allergen Prediction")
st.write("ESM-2 embeddings + CNN + BiLSTM")
st.caption("Research prototype only. Predictions are not a substitute for experimental allergenicity testing.")

def validate_sequence(s):
    seq = "".join(s.split()).upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
    if not seq:
        raise ValueError("Enter a protein amino-acid sequence.")
    bad = sorted(set(seq) - allowed)
    if bad:
        raise ValueError("Unsupported characters: " + ", ".join(bad))
    if len(seq) > 1022:
        raise ValueError("Sequence is longer than 1022 residues; split it into shorter segments.")
    return seq

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None
    # model = tf.keras.models.load_model(MODEL_PATH)
    model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
    )
    scaler = joblib.load(SCALER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model.eval().to(device)
    return model, scaler, esm_model, alphabet, device

def embed_one(seq, esm_model, alphabet, device):
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("query", seq)])
    tokens = tokens.to(device)
    with torch.no_grad():
        reps = esm_model(tokens, repr_layers=[6], return_contacts=False)["representations"][6]
    vec = reps[0, 1:len(seq)+1].mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return vec.reshape(1, -1)

artifacts = load_artifacts()
if artifacts is None:
    st.error("Trained model files are missing. Run `python train.py` successfully first; the app will not generate fake predictions.")
    st.stop()

sequence = st.text_area("Protein sequence", placeholder="Paste amino-acid sequence, e.g. MKK... ", height=150)
if st.button("Predict allergenicity", type="primary"):
    try:
        seq = validate_sequence(sequence)
        model, scaler, esm_model, alphabet, device = artifacts
        X = scaler.transform(embed_one(seq, esm_model, alphabet, device)).reshape(1, 320, 1)
        probs = model.predict(X, verbose=0)[0]
        pred = int(np.argmax(probs))
        label = "Likely allergen" if pred == 1 else "Likely non-allergen"
        st.subheader(label)
        st.metric("Model score for predicted class", f"{float(probs[pred]) * 100:.1f}%")
        st.write({
            "non_allergen_score": round(float(probs[0]), 4),
            "allergen_score": round(float(probs[1]), 4)
        })
        st.warning("These scores are model outputs, not calibrated clinical probabilities. Validate against independent data before scientific use.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

if META_PATH.exists():
    with st.expander("Model information"):
        st.json(json.loads(META_PATH.read_text(encoding="utf-8")))
