# Protein Allergen Prediction — ESM-2 + CNN + BiLSTM

A Streamlit research prototype that predicts whether a protein sequence is likely to be allergenic using ESM-2 (`esm2_t6_8M_UR50D`) embeddings and a trained CNN + BiLSTM classifier.

> **Research use only.** Scores are model outputs, not calibrated clinical probabilities. Do not use this app to make food-safety, medical, regulatory, or clinical decisions. Predictions require independent scientific validation.

## Included
- `app.py` — Streamlit user interface and inference
- `train.py` — training script
- `models/allergen_cnn_bilstm.keras` — trained model
- `models/scaler.joblib` — fitted feature scaler
- `models/metadata.json` — model details and held-out test metrics

## Run locally (Windows PowerShell)

Use Python 3.11 in a clean environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

The first launch downloads the ESM-2 pretrained weights, so it needs internet access and can take some time. On Windows, if Streamlit reports a permission issue writing its user configuration, try:

```powershell
New-Item -ItemType Directory -Force .\.streamlit
$env:STREAMLIT_CONFIG_DIR = "$PWD\.streamlit"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push the contents of this folder to a GitHub repository. The model files are included; do not upload your private training dataset.
2. Open Streamlit Community Cloud and create a new app from that repository.
3. Set the branch to `main` and the main file path to `app.py`.
4. Select Python 3.11 if the deployment settings offer a Python version selector.
5. Deploy. Initial startup downloads the ESM-2 weights and may take several minutes.

The dependency versions are pinned for a more reproducible Python 3.11 deployment. Hosted dependency builds can still change or fail, so check the deployment logs if the app does not start.

## Model information

The saved metadata reports the following held-out split results: accuracy 92.45%, sensitivity 94.70%, specificity 90.13%, MCC 0.850, balanced accuracy 92.41%, ROC-AUC 0.977. These are the values reported by the training run, **not independently verified results**. The metadata itself warns that sequence-homology leakage has not been ruled out; performance may be overestimated if related sequences appear across train/test splits.

The model uses mean-pooled 320-dimensional ESM-2 embeddings, then applies Conv1D and BiLSTM layers over the embedding dimensions. This is not residue-level CNN/BiLSTM processing.

## Training a new model

The dataset must contain `sequence` and `label` columns, with labels `0=non-allergen` and `1=allergen`:

```powershell
python train.py --data "path\to\your_dataset.xlsx" --epochs 60
```

Do not commit private or proprietary datasets. Generated model artifacts are written to `models/`.
