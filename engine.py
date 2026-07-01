import logging
from typing import Dict, Any
import pandas as pd

from model_pipeline import StackingPredictor
from recommender import FoodRecommender
from config import Config

logger = logging.getLogger(__name__)


class GlucoseEngine:
    """
    Orchestrates: pretrained stacking ensemble (LSTM+TabNet+XGBoost) -> risk -> food recs.
    """
    def __init__(self):
        self.predictor = StackingPredictor()
        self.recommender = FoodRecommender(
            excel_path=Config.EXCEL_PATH,
            sensitivity=Config.DEFAULT_SENSITIVITY
        )

    def _classify_risk(self, current: float, slope: float) -> Dict[str, Any]:
        trend = "RISING" if slope > 1 else "FALLING" if slope < -1 else "STABLE"
        risk_type = (
            "HYPOGLYCEMIA" if current < 70 else
            "HYPERGLYCEMIA" if current > 180 else
            "NORMAL"
        )
        return {"trend": trend, "trend_slope": round(slope, 3), "type": risk_type}

    def run(
        self,
        entries: pd.DataFrame,
        carbs_limit: int = Config.DEFAULT_CARBS_LIMIT,
        meal_type: str = "regular"
    ) -> Dict[str, Any]:
        if entries is None or len(entries) == 0:
            raise ValueError("At least one CGM entry is required")

        preds = self.predictor.predict(entries)

        current = float(entries["Glucose"].iloc[-1])
        first = float(entries["Glucose"].iloc[0])
        slope = (current - first) / max(len(entries) - 1, 1)
        risk = self._classify_risk(current, slope)

        recommendation = self.recommender.recommend(
            risk=risk,
            predictions=preds,
            uncertainty={},
            current_glucose=current,
            carbs_limit=carbs_limit,
            meal_type=meal_type
        )

        return {
            "trend": risk,
            "predictions": preds,
            "recommendation": recommendation
        }

    def set_sensitivity(self, s: float):
        self.recommender.set_sensitivity(s)
