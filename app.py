import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from pytorch_tabnet.tab_model import TabNetRegressor

# ==========================
# CONFIG PATH
# ==========================
MODEL_DIR = "CGM_LSTM_TabNet_XGBoost_Stacking"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# LOAD SCALERS
# ==========================
feature_scaler = joblib.load(os.path.join(MODEL_DIR, "stack_feature_scaler.pkl"))
glucose_scaler = joblib.load(os.path.join(MODEL_DIR, "stack_glucose_scaler.pkl"))
meta_scaler = joblib.load(os.path.join(MODEL_DIR, "stack_meta_scaler.pkl"))

# ==========================
# LSTM MODEL (SAME ARCH)
# ==========================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])


# ==========================
# LOAD LSTM
# ==========================
input_size = 10  # change if your feature size differs

lstm_model = LSTMPredictor(input_size=input_size)
lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "stack_lstm_best.pth"), map_location=DEVICE))
lstm_model.to(DEVICE)
lstm_model.eval()

# ==========================
# LOAD TABNET MODELS
# ==========================
tabnet_30 = TabNetRegressor()
tabnet_30.load_model(os.path.join(MODEL_DIR, "stack_tabnet_30min.zip"))

tabnet_60 = TabNetRegressor()
tabnet_60.load_model(os.path.join(MODEL_DIR, "stack_tabnet_60min.zip"))

tabnet_120 = TabNetRegressor()
tabnet_120.load_model(os.path.join(MODEL_DIR, "stack_tabnet_120min.zip"))

# ==========================
# LOAD META MODELS (XGBOOST / RF)
# ==========================
meta_30 = joblib.load(os.path.join(MODEL_DIR, "stack_meta_xgb_30min.pkl"))
meta_60 = joblib.load(os.path.join(MODEL_DIR, "stack_meta_xgb_60min.pkl"))
meta_120 = joblib.load(os.path.join(MODEL_DIR, "stack_meta_xgb_120min.pkl"))

# ==========================
# FEATURE ENGINEERING
# ==========================
def extract_features(cgm):
    cgm = np.array(cgm)

    return {
        "last": cgm[-1],
        "mean": np.mean(cgm),
        "std": np.std(cgm),
        "slope": cgm[-1] - cgm[-3],
        "max": np.max(cgm),
        "min": np.min(cgm),
        "range": np.max(cgm) - np.min(cgm),
        "cv": np.std(cgm) / (np.mean(cgm) + 1e-6),
        "var": np.mean(np.abs(np.diff(cgm)))
    }


def build_meta(lstm_pred, tabnet_pred, feat):
    return np.column_stack([
        lstm_pred,
        tabnet_pred,
        feat["last"],
        feat["mean"],
        feat["std"],
        feat["slope"],
        feat["max"],
        feat["min"],
        feat["range"],
        feat["cv"],
        feat["var"]
    ])


# ==========================
# PREDICTION PIPELINE
# ==========================
def predict(cgm_input):

    # reshape for LSTM
    x = np.array(cgm_input).reshape(1, -1, 1)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    # -------------------
    # 1. LSTM prediction
    # -------------------
    with torch.no_grad():
        lstm_pred = lstm_model(x_tensor).cpu().numpy()

    # -------------------
    # 2. TabNet prediction
    # -------------------
    flat = np.array(cgm_input).reshape(1, -1)

    tabnet_pred = np.column_stack([
        tabnet_30.predict(flat),
        tabnet_60.predict(flat),
        tabnet_120.predict(flat)
    ])

    # -------------------
    # 3. Meta features
    # -------------------
    feat = extract_features(cgm_input)
    meta = build_meta(lstm_pred, tabnet_pred, feat)

    meta_scaled = meta_scaler.transform(meta)

    # -------------------
    # 4. XGBoost fusion
    # -------------------
    p30 = meta_30.predict(meta_scaled)[0]
    p60 = meta_60.predict(meta_scaled)[0]
    p120 = meta_120.predict(meta_scaled)[0]

    return {
        "30min": p30,
        "60min": p60,
        "120min": p120
    }


# ==========================
# RISK ENGINE (your CDSS layer)
# ==========================
def risk_engine(preds, current):

    peak = max(preds.values())
    trough = min(preds.values())

    risk = "LOW"

    if current < 70 or trough < 70:
        risk = "HIGH (HYPO)"
    elif current > 180 or peak > 180:
        risk = "HIGH (HYPER)"
    elif abs(preds["30min"] - current) > 25:
        risk = "MEDIUM"

    return risk


# ==========================
# STREAMLIT UI
# ==========================
st.title("🧠 CGM Stacking CDSS (LSTM + TabNet + XGBoost)")

st.write("Enter 10 CGM values:")

cgm_input = st.text_input("CGM values (comma separated)", "140,145,150,155,160,165,170,175,180,185")

if st.button("Run Prediction"):

    try:
        cgm = [float(x.strip()) for x in cgm_input.split(",")]

        if len(cgm) != 10:
            st.error("Please enter exactly 10 values")
        else:

            preds = predict(cgm)
            current = cgm[-1]

            risk = risk_engine(preds, current)

            st.subheader("📊 Predictions")
            st.write(preds)

            st.subheader("⚠️ Risk Level")
            st.write(risk)

    except Exception as e:
        st.error(f"Error: {str(e)}")
