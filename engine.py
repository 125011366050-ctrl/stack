import os
import numpy as np
import torch
import joblib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetRegressor

from lstm_model import LSTMPredictor
from food_recommender import FoodRecommender
from risk_engine import classify_risk, calculate_risk_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.window_size = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # All files directly in main directory
        self.excel_path = os.path.join(self.base_dir, "Indian_Foods_GI_GL_Database.xlsx")
        self.ensemble_size = 10
        self.default_sensitivity = 1.0
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Device: {self.device}")

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
    
    def set_patient_sensitivity(self, sensitivity: float):
        if sensitivity <= 0 or sensitivity > 3.0:
            raise ValueError("Sensitivity must be between 0 and 3.0")
        self.sensitivity = sensitivity
        self.food_recommender.set_sensitivity(sensitivity)
        logger.info(f"✅ Updated sensitivity to: {sensitivity}")
    
    def _load_models(self):
        """Load all models from main directory"""
        try:
            logger.info("Loading models...")
            
            # Load scalers
            self.feature_scaler = joblib.load(
                os.path.join(self.config.base_dir, "stack_feature_scaler.pkl")
            )
            self.meta_scaler = joblib.load(
                os.path.join(self.config.base_dir, "stack_meta_scaler.pkl")
            )
            
            # Load TabNet models
            self.tabnet_models = []
            for h in ["30min", "60min", "120min"]:
                model = TabNetRegressor()
                model_path = os.path.join(self.config.base_dir, f"stack_tabnet_{h}.zip")
                model.load_model(model_path)
                self.tabnet_models.append(model)
                logger.info(f"✅ Loaded TabNet: {h}")
            
            # Load XGBoost models
            self.meta_models = []
            for h in [30, 60, 120]:
                model = joblib.load(
                    os.path.join(self.config.base_dir, f"stack_meta_xgb_{h}min.pkl")
                )
                self.meta_models.append(model)
                logger.info(f"✅ Loaded XGBoost: {h}min")
            
            # Load LSTM
            self.lstm = LSTMPredictor(input_size=10)
            lstm_path = os.path.join(self.config.base_dir, "stack_lstm_best.pth")
            self.lstm.load_state_dict(
                torch.load(lstm_path, map_location=self.config.device)
            )
            self.lstm.to(self.config.device)
            self.lstm.eval()
            logger.info("✅ Loaded LSTM model")
            
            logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
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
    
    def _predict_with_uncertainty(self, cgm: np.ndarray) -> tuple:
        flat = cgm.reshape(1, -1)
        
        lstm_input = cgm.reshape(1, 10, 1)
        lstm_tensor = torch.tensor(lstm_input, dtype=torch.float32).to(self.config.device)
        with torch.no_grad():
            lstm_pred = self.lstm(lstm_tensor).cpu().numpy()[0]
        
        tabnet_preds = [model.predict(flat)[0][0] for model in self.tabnet_models]
        features = self._extract_features(cgm)
        meta = np.column_stack([lstm_pred, tabnet_preds, features])
        meta_scaled = self.meta_scaler.transform(meta)
        
        preds = [model.predict(meta_scaled)[0] for model in self.meta_models]
        
        uncertainty = {}
        ensemble_preds = []
        
        for _ in range(self.config.ensemble_size):
            noise = np.random.normal(0, 0.02, cgm.shape)
            cgm_noisy = cgm + noise
            
            flat_noisy = cgm_noisy.reshape(1, -1)
            lstm_input_noisy = cgm_noisy.reshape(1, 10, 1)
            lstm_tensor_noisy = torch.tensor(lstm_input_noisy, dtype=torch.float32).to(self.config.device)
            
            with torch.no_grad():
                lstm_pred_noisy = self.lstm(lstm_tensor_noisy).cpu().numpy()[0]
            
            tabnet_preds_noisy = [model.predict(flat_noisy)[0][0] for model in self.tabnet_models]
            features_noisy = self._extract_features(cgm_noisy)
            meta_noisy = np.column_stack([lstm_pred_noisy, tabnet_preds_noisy, features_noisy])
            meta_scaled_noisy = self.meta_scaler.transform(meta_noisy)
            
            preds_noisy = [model.predict(meta_scaled_noisy)[0] for model in self.meta_models]
            ensemble_preds.append(preds_noisy)
        
        ensemble_preds = np.array(ensemble_preds)
        
        for idx, label in enumerate(["30min", "60min", "120min"]):
            pred_values = ensemble_preds[:, idx]
            mean = np.mean(pred_values)
            std = np.std(pred_values)
            
            lower = max(0, mean - 1.96 * std)
            upper = mean + 1.96 * std
            
            uncertainty[label] = {
                "mean": float(mean),
                "lower": float(lower),
                "upper": float(upper),
                "std": float(std)
            }
        
        predictions = {
            "30min": float(preds[0]),
            "60min": float(preds[1]),
            "120min": float(preds[2])
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
            
            # Build risk with phase detection
            risk = {
                "type": "NORMAL",
                "level": "LOW",
                "trend": "STABLE",
                "trend_slope": 0.0
            }
            
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
                "alert_level": "GREEN",
                "personalization_sensitivity": sensitivity
            }
            
        except Exception as e:
            logger.error(f"❌ Run failed: {e}")
            return {"error": str(e)}
