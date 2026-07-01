import logging
from typing import Dict, Any, List, Optional

from recommender import FoodRecommender
from config import Config

logger = logging.getLogger(__name__)


class GlucoseEngine:
    """
    Lightweight CDSS orchestration layer optimized for Streamlit deployment.
    """

    def __init__(self):
        # DO NOT load heavy objects repeatedly in Streamlit
        self.recommender = FoodRecommender(
            excel_path=Config.EXCEL_PATH,
            sensitivity=Config.DEFAULT_SENSITIVITY
        )

    # ==============================
    # TREND ANALYSIS
    # ==============================
    def _analyze_trend(
        self,
        glucose_history: List[float]
    ) -> Dict[str, Any]:

        if len(glucose_history) < 2:
            return {"trend": "STABLE", "trend_slope": 0.0, "type": "NORMAL"}

        dt = (len(glucose_history) - 1) * 5
        dt = max(dt, 1)

        slope = (glucose_history[-1] - glucose_history[0]) / dt

        trend = (
            "RISING" if slope > 1 else
            "FALLING" if slope < -1 else
            "STABLE"
        )

        current = glucose_history[-1]

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

    # ==============================
    # SIMPLE PREDICTION
    # ==============================
    def _predict(self, current: float, slope: float) -> Dict[str, float]:
        return {
            "30min": round(current + slope * 30, 1),
            "60min": round(current + slope * 60, 1),
            "120min": round(current + slope * 120, 1),
        }

    # ==============================
    # UNCERTAINTY ESTIMATION
    # ==============================
    def _uncertainty(
        self,
        preds: Dict[str, float],
        history: List[float]
    ) -> Dict[str, Dict[str, float]]:

        if len(history) > 1:
            volatility = sum(
                abs(history[i] - history[i - 1])
                for i in range(1, len(history))
            ) / (len(history) - 1)
        else:
            volatility = 5.0

        m = max(5.0, volatility)

        return {
            "30min": {"lower": preds["30min"] - m, "upper": preds["30min"] + m},
            "60min": {"lower": preds["60min"] - 1.5 * m, "upper": preds["60min"] + 1.5 * m},
            "120min": {"lower": preds["120min"] - 2 * m, "upper": preds["120min"] + 2 * m},
        }

    # ==============================
    # MAIN PIPELINE
    # ==============================
    def run(
        self,
        glucose_history: List[float],
        carbs_limit: int = Config.DEFAULT_CARBS_LIMIT,
        meal_type: str = "regular"
    ) -> Dict[str, Any]:

        if not glucose_history:
            raise ValueError("glucose_history required")

        current = glucose_history[-1]

        trend = self._analyze_trend(glucose_history)
        preds = self._predict(current, trend["trend_slope"])
        unc = self._uncertainty(preds, glucose_history)

        recommendation = self.recommender.recommend(
            risk=trend,
            predictions=preds,
            uncertainty=unc,
            current_glucose=current,
            carbs_limit=carbs_limit,
            meal_type=meal_type
        )

        return {
            "trend": trend,
            "predictions": preds,
            "uncertainty": unc,
            "recommendation": recommendation
        }

    def set_sensitivity(self, s: float):
        self.recommender.set_sensitivity(s)
