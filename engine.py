import os
import numpy as np
import torch
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor
from typing import List, Dict, Any


# ==============================
# CONFIG
# ==============================
class Config:
    def __init__(self):
        self.window_size = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = "CGM_LSTM_TabNet_XGBoost_Stacking"


# ==============================
# ENGINE
# ==============================
class ClinicalOrchestrator:
    def __init__(self, config: Config):
        self.config = config

        # ---------- LOAD SCALERS ----------
        self.feature_scaler = joblib.load(
            os.path.join(config.model_dir, "stack_feature_scaler.pkl")
        )
        self.meta_scaler = joblib.load(
            os.path.join(config.model_dir, "stack_meta_scaler.pkl")
        )
        self.glucose_scaler = joblib.load(
            os.path.join(config.model_dir, "stack_glucose_scaler.pkl")
        )

        # ---------- LOAD FEATURE NAMES ----------
        self.meta_feature_names = joblib.load(
            os.path.join(config.model_dir, "stack_meta_feature_names.pkl")
        )

        # ---------- LOAD TABNET MODELS ----------
        self.tabnet_models = []
        for h in ["30min", "60min", "120min"]:
            model = TabNetRegressor()
            model.load_model(os.path.join(config.model_dir, f"stack_tabnet_{h}"))
            self.tabnet_models.append(model)

        # ---------- LOAD XGBOOST META MODELS ----------
        self.meta_models = []
        for h in [30, 60, 120]:
            model = joblib.load(
                os.path.join(config.model_dir, f"stack_meta_xgb_{h}min.pkl")
            )
            self.meta_models.append(model)

        # ---------- LOAD LSTM ----------
        from lstm_model import LSTMPredictor  # you must have same class

        self.lstm = LSTMPredictor(input_size=1)
        self.lstm.load_state_dict(
            torch.load(
                os.path.join(config.model_dir, "stack_lstm_best.pth"),
                map_location=config.device
            )
        )
        self.lstm.to(config.device)
        self.lstm.eval()

    # ==============================
    # MAIN RUN
    # ==============================
    def run(
        self,
        cgm_readings: List[float],
        carbs: int,
        protein: int,
        fat: int
    ) -> Dict[str, Any]:

        if len(cgm_readings) < 10:
            return {"error": "Need exactly 10 CGM values"}

        # ==============================
        # 1. PREPROCESS INPUT
        # ==============================
        cgm = np.array(cgm_readings).reshape(1, -1)

        # ==============================
        # 2. LSTM INPUT (3 predictions)
        # ==============================
        lstm_input = cgm.reshape(1, 10, 1)
        lstm_out = self._predict_lstm(lstm_input)

        # ==============================
        # 3. TABNET INPUT (flattened)
        # ==============================
        tabnet_input = cgm.reshape(1, -1)
        tabnet_out = self._predict_tabnet(tabnet_input)

        # ==============================
        # 4. CGM FEATURES (meta features)
        # ==============================
        cgm_features = self._extract_features(cgm)

        # ==============================
        # 5. BUILD META FEATURES
        # ==============================
        meta_features = np.column_stack([
            lstm_out,
            tabnet_out,
            cgm_features
        ])

        meta_features = self.meta_scaler.transform(meta_features)

        # ==============================
        # 6. XGBOOST FUSION
        # ==============================
        preds = []
        for i, model in enumerate(self.meta_models):
            preds.append(model.predict(meta_features)[0])

        preds = np.array(preds)

        # ==============================
        # 7. RISK ENGINE
        # ==============================
        risk = self._risk(preds)

        # ==============================
        # 8. RULE-BASED CDSS
        # ==============================
        return {
            "current_glucose": float(cgm[0, -1]),
            "predictions": {
                "30min": float(preds[0]),
                "60min": float(preds[1]),
                "120min": float(preds[2])
            },
            "risk": risk,
            "recommendation": self._recommend(risk, carbs, protein, fat)
        }

    # ==============================
    # LSTM PREDICTION
    # ==============================
    def _predict_lstm(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.config.device)
        with torch.no_grad():
            out = self.lstm(x).cpu().numpy()[0]
        return out

    # ==============================
    # TABNET PREDICTION
    # ==============================
    def _predict_tabnet(self, x):
        preds = []
        for model in self.tabnet_models:
            preds.append(model.predict(x)[0][0])
        return np.array(preds)

    # ==============================
    # FEATURE ENGINEERING
    # ==============================
    def _extract_features(self, x):
        g = x[0]

        last = g[-1]
        mean = np.mean(g)
        std = np.std(g)
        slope = g[-1] - g[-3]
        mx = np.max(g)
        mn = np.min(g)

        return np.array([
            last, mean, std, slope, mx, mn,
            mx - mn,
            std / (mean + 1e-6),
            np.mean(np.abs(np.diff(g)))
        ]).reshape(1, -1)

    # ==============================
    # RISK ENGINE
    # ==============================
    def _risk(self, preds):
        peak = np.max(preds)
        trough = np.min(preds)
        current = preds[0]

        if current < 70 or trough < 70:
            return {"level": "HIGH", "type": "HYPOGLYCEMIA"}

        if current > 180 or peak > 180:
            return {"level": "HIGH", "type": "HYPERGLYCEMIA"}

        return {"level": "LOW", "type": "NORMAL"}

    # ==============================
    # RECOMMENDATION
    # ==============================
    def _recommend(self, risk, carbs, protein, fat):

        if risk["level"] == "HIGH":
            return {
                "food": "Low carb + protein meal",
                "activity": "Light walking 10–15 min"
            }

        return {
            "food": "Balanced diet",
            "activity": "Normal activity"
        }
