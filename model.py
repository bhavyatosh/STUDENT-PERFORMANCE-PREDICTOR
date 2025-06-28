# model.py

import gdown
import joblib
import os

# Google Drive file IDs
MODEL_ID = '1cuBYD9LescJJPdKMxeH5MTZV1Ssjqo23'
SCALER_ID = '1ETQI6QQ8ItL6QVLUb7qD95sFBQfArD8T'

# Local file paths
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

def download_if_missing(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

def load_model_and_scaler():
    download_if_missing(MODEL_ID, MODEL_PATH)
    download_if_missing(SCALER_ID, SCALER_PATH)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
