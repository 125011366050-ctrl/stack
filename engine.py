import numpy as np
import logging
from typing import Dict, Any
import pandas as pd

from model_pipeline import StackingPredictor
from recommender import FoodRecommender
from config import Config

logger = logging.getLogger(__name__)


class GlucoseEngine:

    def __init__(self):
        self.predictor = StackingPredictor()
        self.recommender = FoodRecommender(
            excel_path=Config.EXCEL_PATH,
            sensitivity=Config.DEFAULT_SENSITIVITY
        )

        # store residual history for uncertainty estimation
        self.residuals = []

    def _classify_risk(self, current: float, slope: float) -> Dict[str, Any]:
        trend = "RISING" if slope > 1 else "FALLING" if slope < -1 else "STABLE"

        risk_type = (
            "HYPOGLYCEMIA" if current < 70 else
            "HYPERGLYCEMIA" if current > 180 else
            "NORMAL"
        )

        return {
            "trend": trend,
            "trend_slope": round(slope, 3),
            "type": risk_type
        }

    # -----------------------------
    # 🔥 NEW: spike control
    # -----------------------------
    def _clip_physiology(self, prev: float, pred: float, horizon: int) -> float:
        max_jump = {
            30: 40,
            60: 80,
            120: 120
        }.get(horizon, 100)

        return float(np.clip(pred, prev - max_jump, prev + max_jump))

    # -----------------------------
    # 🔥 NEW: uncertainty bounds
    # -----------------------------
    def _compute_bounds(self, pred: float) -> Dict[str, float]:
        if len(self.residuals) > 10:
            std = np.std(self.residuals)
        else:
            std = 10  # fallback

        lower = pred - 1.96 * std
        upper = pred + 1.96 * std

        return {
            "lower": float(max(0, lower)),
            "upper": float(upper)
        }

    def run(
        self,
        entries: pd.DataFrame,
        carbs_limit: int = Config.DEFAULT_CARBS_LIMIT,
        meal_type: str = "regular"
    ) -> Dict[str, Any]:

        if entries is None or len(entries) == 0:
            raise ValueError("At least one CGM entry is required")

        raw_preds = self.predictor.predict(entries)

        current = float(entries["Glucose"].iloc[-1])
        first = float(entries["Glucose"].iloc[0])
        slope = (current - first) / max(len(entries) - 1, 1)

        risk = self._classify_risk(current, slope)

        # -----------------------------
        # 🔥 FIX: spike control + smoothing
        # -----------------------------
        cleaned_preds = {}
        prev = current

        for horizon, pred in raw_preds.items():

            pred = float(pred)

            # 1. physiology clipping
            pred = self._clip_physiology(prev, pred, int(horizon))

            # 2. smoothing (prevents sharp spikes)
            pred = 0.7 * prev + 0.3 * pred

            # store residual for future uncertainty
            self.residuals.append(abs(pred - prev))

            # 3. bounds
            bounds = self._compute_bounds(pred)

            cleaned_preds[horizon] = {
                "prediction": round(pred, 2),
                "lower": round(bounds["lower"], 2),
                "upper": round(bounds["upper"], 2)
            }

            prev = pred

        recommendation = self.recommender.recommend(
            risk=risk,
            predictions=cleaned_preds,
            uncertainty={k: v for k, v in cleaned_preds.items()},
            current_glucose=current,
            carbs_limit=carbs_limit,
            meal_type=meal_type
        )

        return {
            "trend": risk,
            "predictions": cleaned_preds,
            "recommendation": recommendation
        }

    def set_sensitivity(self, s: float):
        self.recommender.set_sensitivity(s)
