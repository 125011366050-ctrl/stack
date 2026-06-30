import os
import numpy as np
import joblib
import torch
import torch.nn as nn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pytorch_tabnet.tab_model import TabNetRegressor

# ==========================
# APP INIT
# ==========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.getcwd()

print("📁 BASE DIR:", BASE_DIR)
print("📂 FILES:", os.listdir(BASE_DIR))

# ==========================
# LOAD META MODELS (XGBOOST)
# ==========================
meta_30 = joblib.load(os.path.join(BASE_DIR, "stack_meta_xgb_30min.pkl"))
meta_60 = joblib.load(os.path.join(BASE_DIR, "stack_meta_xgb_60min.pkl"))
meta_120 = joblib.load(os.path.join(BASE_DIR, "stack_meta_xgb_120min.pkl"))

meta_scaler = joblib.load(os.path.join(BASE_DIR, "stack_meta_scaler.pkl"))

# ==========================
# LOAD SCALERS
# ==========================
feature_scaler = joblib.load(os.path.join(BASE_DIR, "stack_feature_scaler.pkl"))

# ==========================
# LSTM MODEL
# ==========================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])

# ⚠️ CHANGE THIS IF YOUR INPUT SIZE IS DIFFERENT
INPUT_SIZE = 10

lstm_model = LSTMPredictor(INPUT_SIZE).to(DEVICE)
lstm_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "stack_lstm_best.pth"), map_location=DEVICE)
)
lstm_model.eval()

# ==========================
# TABNET MODELS
# ==========================
tabnet_30 = TabNetRegressor()
tabnet_30.load_model(os.path.join(BASE_DIR, "stack_tabnet_30min.zip"))

tabnet_60 = TabNetRegressor()
tabnet_60.load_model(os.path.join(BASE_DIR, "stack_tabnet_60min.zip"))

tabnet_120 = TabNetRegressor()
tabnet_120.load_model(os.path.join(BASE_DIR, "stack_tabnet_120min.zip"))

# ==========================
# FEATURE ENGINEERING
# ==========================
def extract_features(cgm):
    cgm = np.array(cgm)

    return np.array([
        cgm[-1],                          # last
        np.mean(cgm),                     # mean
        np.std(cgm),                      # std
        cgm[-1] - cgm[-3],                # slope
        np.max(cgm),                      # max
        np.min(cgm),                      # min
        np.max(cgm) - np.min(cgm),        # range
        np.std(cgm) / (np.mean(cgm)+1e-6),# cv
        np.mean(np.abs(np.diff(cgm)))     # variability
    ])

# ==========================
# META FEATURE BUILDER
# ==========================
def build_meta(lstm_pred, tabnet_pred, cgm_features):

    return np.column_stack([
        lstm_pred,
        tabnet_pred,
        cgm_features
    ])

# ==========================
# PREDICTION PIPELINE
# ==========================
def predict_pipeline(cgm):

    cgm = np.array(cgm)

    # reshape for LSTM (1, time, features)
    lstm_input = cgm.reshape(1, -1, 1)

    lstm_tensor = torch.tensor(lstm_input, dtype=torch.float32).to(DEVICE)

    # -------------------
    # 1. LSTM prediction
    # -------------------
    with torch.no_grad():
        lstm_pred = lstm_model(lstm_tensor).cpu().numpy()

    # -------------------
    # 2. TabNet prediction
    # -------------------
    flat = cgm.reshape(1, -1)

    tabnet_pred = np.column_stack([
        tabnet_30.predict(flat),
        tabnet_60.predict(flat),
        tabnet_120.predict(flat)
    ])

    # -------------------
    # 3. CGM features
    # -------------------
    cgm_feat = extract_features(cgm)

    # -------------------
    # 4. Meta features
    # -------------------
    meta = build_meta(lstm_pred, tabnet_pred, cgm_feat.reshape(1, -1))

    meta_scaled = meta_scaler.transform(meta)

    # -------------------
    # 5. XGBoost fusion
    # -------------------
    p30 = float(meta_30.predict(meta_scaled)[0])
    p60 = float(meta_60.predict(meta_scaled)[0])
    p120 = float(meta_120.predict(meta_scaled)[0])

    return {
        "30min": p30,
        "60min": p60,
        "120min": p120
    }

# ==========================
# CDSS RISK ENGINE
# ==========================
def risk_engine(preds, current):

    peak = max(preds.values())
    trough = min(preds.values())

    if current < 70 or trough < 70:
        return "HIGH - HYPOGLYCEMIA"
    elif current > 180 or peak > 180:
        return "HIGH - HYPERGLYCEMIA"
    elif abs(preds["30min"] - current) > 25:
        return "MEDIUM - TREND RISK"
    else:
        return "LOW RISK"

# ==========================
# API ENDPOINTS
# ==========================
@app.get("/")
def home():
    return {
        "status": "CDSS API Running 🚀",
        "model": "LSTM + TabNet + XGBoost Stacking"
    }

@app.post("/predict")
def predict(data: dict):

    try:
        cgm = data["cgm"]

        if len(cgm) != 10:
            return {"error": "Please send exactly 10 CGM values"}

        preds = predict_pipeline(cgm)

        return {
            "input_current": cgm[-1],
            "predictions": preds,
            "risk": risk_engine(preds, cgm[-1])
        }

    except Exception as e:
        return {"error": str(e)}
