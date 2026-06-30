import os
import numpy as np
import joblib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import torch - fallback if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch not available - using fallback mode")

# Try to import TabNet
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
    logger.info("✅ TabNet available")
except ImportError:
    TABNET_AVAILABLE = False
    logger.warning("⚠️ TabNet not available")

from food_recommender import FoodRecommender

# LSTM model (only if torch available)
if TORCH_AVAILABLE:
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=10, hidden_size=128, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 3)

        def forward(self, x):
            out, (h, c) = self.lstm(x)
            return self.fc(h[-1])
else:
    # Fallback LSTM class
    class LSTMPredictor:
        def __init__(self, *args, **kwargs):
            pass

class Config:
    def __init__(self):
        self.window_size = 10
        self.device = "cpu"
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.excel_path = os.path.join(self.base_dir, "Indian_Foods_GI_GL_Database.xlsx")
        self.ensemble_size = 5
        self.default_sensitivity = 1.0
        logger.info(f"Base directory: {self.base_dir}")

class ClinicalOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.sensitivity = config.default_sensitivity
        self.food_recommender = FoodRecommender(
            config.excel_path,
            sensitivity=self.sensitivity
        )
        self._load_models()
        logger.info(f"✅ ClinicalOrchestrator initialized")
        logger.info(f"Torch available: {TORCH_AVAILABLE}")
        logger.info(f"TabNet available: {TABNET_AVAILABLE}")
    
    def set_patient_sensitivity(self, sensitivity: float):
        if sensitivity <= 0 or sensitivity > 3.0:
            raise ValueError("Sensitivity must be between 0 and 3.0")
        self.sensitivity = sensitivity
        self.food_recommender.set_sensitivity(sensitivity)
        logger.info(f"✅ Updated sensitivity to: {sensitivity}")
    
    def _load_models(self):
        """Load all models - with graceful fallback"""
        try:
            logger.info("Loading models...")
            
            # Load scalers (always available)
            self.feature_scaler = joblib.load(
                os.path.join(self.config.base_dir, "stack_feature_scaler.pkl")
            )
            self.meta_scaler = joblib.load(
                os.path.join(self.config.base_dir, "stack_meta_scaler.pkl")
            )
            logger.info("✅ Loaded scalers")
            
            # Load TabNet models if available
            self.tabnet_models = []
            if TABNET_AVAILABLE:
                for h in ["30min", "60min", "120min"]:
                    try:
                        model = TabNetRegressor()
                        model_path = os.path.join(self.config.base_dir, f"stack_tabnet_{h}.zip")
                        model.load_model(model_path)
                        self.tabnet_models.append(model)
                        logger.info(f"✅ Loaded TabNet: {h}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load TabNet {h}: {e}")
            
            # Load XGBoost models
            self.meta_models = []
            for h in [30, 60, 120]:
                try:
                    model = joblib.load(
                        os.path.join(self.config.base_dir, f"stack_meta_xgb_{h}min.pkl")
                    )
                    self.meta_models.append(model)
                    logger.info(f"✅ Loaded XGBoost: {h}min")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load XGBoost {h}: {e}")
            
            # Load LSTM if torch available
            if TORCH_AVAILABLE:
                try:
                    self.lstm = LSTMPredictor(input_size=10)
                    lstm_path = os.path.join(self.config.base_dir, "stack_lstm_best.pth")
                    self.lstm.load_state_dict(
                        torch.load(lstm_path, map_location='cpu')
                    )
                    self.lstm.eval()
                    logger.info("✅ Loaded LSTM model")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load LSTM: {e}")
                    self.lstm = None
            else:
                self.lstm = None
            
            logger.info("✅ Models loaded (with fallbacks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            # Continue without models - use fallback predictions
    
    def _extract_features(self, cgm: np.ndarray) -> np.ndarray:
        g = cgm.flatten()
        return np.array([
            g[-1], np.mean(g), np.std(g),
            g[-1] - g[-3] if len(g) >= 3 else 0,
            np.max(g), np.min(g),
            np.max(g) - np.min(g),
            np.std(g) / (np.mean(g) + 1e-6),
            np.mean(np.abs(np.diff(g)))
        ]).reshape(1, -1)
    
    def _predict_fallback(self, cgm: np.ndarray) -> Dict[str, float]:
        """Fallback prediction when models are not available"""
        g = cgm.flatten()
        current = g[-1]
        # Simple linear extrapolation
        slope = (g[-1] - g[-5]) / 4 if len(g) >= 5 else 0
        return {
            "30min": float(current + slope * 1),
            "60min": float(current + slope * 2),
            "120min": float(current + slope * 4)
        }
    
    def _predict_with_uncertainty(self, cgm: np.ndarray) -> tuple:
        try:
            # Try using models if available
            if hasattr(self, 'lstm') and self.lstm is not None:
                # Use full model prediction
                flat = cgm.reshape(1, -1)
                
                # LSTM
                lstm_input = cgm.reshape(1, 10, 1)
                lstm_tensor = torch.tensor(lstm_input, dtype=torch.float32)
                with torch.no_grad():
                    lstm_pred = self.lstm(lstm_tensor).cpu().numpy()[0]
                
                # TabNet
                tabnet_preds = []
                if self.tabnet_models:
                    for model in self.tabnet_models:
                        tabnet_preds.append(model.predict(flat)[0][0])
                else:
                    tabnet_preds = [0, 0, 0]
                
                # Features
                features = self._extract_features(cgm)
                meta = np.column_stack([lstm_pred, tabnet_preds, features])
                meta_scaled = self.meta_scaler.transform(meta)
                
                # XGBoost
                if self.meta_models:
                    preds = [model.predict(meta_scaled)[0] for model in self.meta_models]
                else:
                    preds = [float(cgm[0, -1]), float(cgm[0, -1]), float(cgm[0, -1])]
                
                # Uncertainty estimation
                uncertainty = {}
                for idx, label in enumerate(["30min", "60min", "120min"]):
                    mean = float(preds[idx])
                    std = max(1.0, mean * 0.05)  # 5% uncertainty
                    uncertainty[label] = {
                        "mean": mean,
                        "lower": max(0, mean - 1.96 * std),
                        "upper": mean + 1.96 * std,
                        "std": std
                    }
                
                predictions = {
                    "30min": float(preds[0]),
                    "60min": float(preds[1]),
                    "120min": float(preds[2])
                }
                
                return predictions, uncertainty
        except Exception as e:
            logger.warning(f"⚠️ Model prediction failed, using fallback: {e}")
        
        # Fallback prediction
        predictions = self._predict_fallback(cgm)
        uncertainty = {}
        for label in ["30min", "60min", "120min"]:
            mean = predictions[label]
            std = max(1.0, mean * 0.1)
            uncertainty[label] = {
                "mean": mean,
                "lower": max(0, mean - 1.96 * std),
                "upper": mean + 1.96 * std,
                "std": std
            }
        
        return predictions, uncertainty
    
    def run(
        self,
        cgm_readings: List[float],
        carbs: int = 30,
        meal_type: str = "regular",
        patient_sensitivity: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            sensitivity = patient_sensitivity if patient_sensitivity is not None else self.sensitivity
            
            if len(cgm_readings) != 10:
                return {"error": f"Need exactly 10 CGM values, got {len(cgm_readings)}"}
            
            cgm = np.array(cgm_readings).reshape(1, -1)
            current_glucose = float(cgm[0, -1])
            
            predictions, uncertainty = self._predict_with_uncertainty(cgm)
            
            # Simple risk detection
            p30 = predictions.get("30min", current_glucose)
            trend = "STABLE"
            if p30 > current_glucose + 5:
                trend = "RISING"
            elif p30 < current_glucose - 5:
                trend = "FALLING"
            
            risk = {
                "type": "NORMAL",
                "level": "LOW",
                "trend": trend,
                "trend_slope": float(p30 - current_glucose)
            }
            
            # Check for hypo/hyper
            if current_glucose < 70 or min(predictions.values()) < 70:
                risk["type"] = "HYPOGLYCEMIA"
                risk["level"] = "HIGH"
            elif current_glucose > 180 or max(predictions.values()) > 180:
                risk["type"] = "HYPERGLYCEMIA"
                risk["level"] = "HIGH"
            
            food_recommendation = self.food_recommender.recommend(
                risk=risk,
                predictions=predictions,
                uncertainty=uncertainty,
                current_glucose=current_glucose,
                carbs_limit=carbs,
                meal_type=meal_type
            )
            
            return {
                "current_glucose": current_glucose,
                "predictions": predictions,
                "uncertainty": uncertainty,
                "risk": risk,
                "food_recommendation": food_recommendation,
                "timestamp": datetime.now().isoformat(),
                "alert_level": "RED" if risk["level"] == "HIGH" else "GREEN",
                "personalization_sensitivity": sensitivity,
                "models_available": {
                    "torch": TORCH_AVAILABLE,
                    "tabnet": TABNET_AVAILABLE
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Run failed: {e}")
            return {"error": str(e)}
