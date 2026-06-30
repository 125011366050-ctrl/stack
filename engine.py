import logging
from typing import Dict, Any, List, Optional

from recommender import FoodRecommender
from config import Config

logger = logging.getLogger(__name__)


class GlucoseEngine:
    """
    Orchestration layer for the CDSS pipeline:
    raw glucose readings -> trend/risk analysis -> prediction + uncertainty -> food recommendation

    NOTE: The prediction and uncertainty logic here is a placeholder structure
    (simple linear extrapolation + heuristic bounds). If you already have a trained
    forecasting model (e.g. LSTM/ARIMA/Kalman filter), swap out `_predict()` and
    `_estimate_uncertainty()` with calls to that model. This class does NOT replace
    clinical validation of the underlying forecasting approach.
    """

    def __init__(
        self,
        excel_path: str = Config.EXCEL_PATH,
        sensitivity: float = Config.DEFAULT_SENSITIVITY
    ):
        try:
            self.recommender = FoodRecommender(
                excel_path=excel_path,
                sensitivity=sensitivity
            )
        except Exception as e:
            logger.error(f"❌ Engine failed to initialize recommender: {e}")
            raise

    # ==============================
    # TREND ANALYSIS
    # ==============================
    def _analyze_trend(self, glucose_history: List[float], timestamps_min: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Determine trend direction and slope from recent glucose history.
        glucose_history: list of glucose readings, oldest -> newest
        timestamps_min: optional list of minute offsets matching glucose_history
        """
        if not glucose_history or len(glucose_history) < 2:
            return {"trend": "STABLE", "trend_slope": 0.0, "type": "NORMAL"}

        if timestamps_min and len(timestamps_min) == len(glucose_history):
            dt = timestamps_min[-1] - timestamps_min[0]
        else:
            # assume uniform 5-minute spacing if no timestamps given
            dt = (len(glucose_history) - 1) * 5

        dt = dt if dt != 0 else 1
        slope = (glucose_history[-1] - glucose_history[0]) / dt  # mg/dL per minute

        if slope > 1.0:
            trend = "RISING"
        elif slope < -1.0:
            trend = "FALLING"
        else:
            trend = "STABLE"

        current = glucose_history[-1]
        if current < 70:
            risk_type = "HYPOGLYCEMIA"
        elif current > 180:
            risk_type = "HYPERGLYCEMIA"
        else:
            risk_type = "NORMAL"

        return {
            "trend": trend,
            "trend_slope": round(slope, 3),
            "type": risk_type
        }

    # ==============================
    # PREDICTION (placeholder — swap with real model)
    # ==============================
    def _predict(self, current_glucose: float, trend_slope: float) -> Dict[str, float]:
        """
        Simple linear extrapolation placeholder.
        Replace with a trained model's output if available.
        """
        return {
            "30min": round(current_glucose + trend_slope * 30, 1),
            "60min": round(current_glucose + trend_slope * 60, 1),
            "120min": round(current_glucose + trend_slope * 120, 1),
        }

    def _estimate_uncertainty(
        self,
        predictions: Dict[str, float],
        glucose_history: List[float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Heuristic uncertainty bounds based on historical volatility.
        Wider bounds further out in time, and wider if recent readings are noisy.
        Replace with real model confidence intervals if available.
        """
        if glucose_history and len(glucose_history) >= 2:
            diffs = [abs(glucose_history[i] - glucose_history[i - 1]) for i in range(1, len(glucose_history))]
            volatility = sum(diffs) / len(diffs)
        else:
            volatility = 5.0

        base_margin = max(5.0, volatility)

        return {
            "30min": {
                "lower": round(predictions["30min"] - base_margin * 1.0, 1),
                "upper": round(predictions["30min"] + base_margin * 1.0, 1),
            },
            "60min": {
                "lower": round(predictions["60min"] - base_margin * 1.5, 1),
                "upper": round(predictions["60min"] + base_margin * 1.5, 1),
            },
            "120min": {
                "lower": round(predictions["120min"] - base_margin * 2.0, 1),
                "upper": round(predictions["120min"] + base_margin * 2.0, 1),
            },
        }

    # ==============================
    # MAIN PIPELINE ENTRY POINT
    # ==============================
    def run(
        self,
        glucose_history: List[float],
        timestamps_min: Optional[List[float]] = None,
        carbs_limit: int = Config.DEFAULT_CARBS_LIMIT,
        meal_type: str = "regular"
    ) -> Dict[str, Any]:
        """
        Full pipeline: history -> trend -> prediction -> uncertainty -> recommendation
        """
        if not glucose_history:
            raise ValueError("glucose_history must contain at least one reading")

        current_glucose = glucose_history[-1]

        risk = self._analyze_trend(glucose_history, timestamps_min)
        predictions = self._predict(current_glucose, risk["trend_slope"])
        uncertainty = self._estimate_uncertainty(predictions, glucose_history)

        logger.info(
            f"Engine run: current={current_glucose}, trend={risk['trend']}, "
            f"slope={risk['trend_slope']}, type={risk['type']}"
        )

        recommendation = self.recommender.recommend(
            risk=risk,
            predictions=predictions,
            uncertainty=uncertainty,
            current_glucose=current_glucose,
            carbs_limit=carbs_limit,
            meal_type=meal_type
        )

        return {
            "input": {
                "glucose_history": glucose_history,
                "current_glucose": current_glucose
            },
            "trend_analysis": risk,
            "predictions": predictions,
            "uncertainty": uncertainty,
            "recommendation": recommendation
        }

    def set_sensitivity(self, sensitivity: float):
        self.recommender.set_sensitivity(sensitivity)
