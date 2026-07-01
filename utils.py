import logging
from typing import Dict, Any
from config import Config


# ==========================
# LOGGING
# ==========================
def setup_logging():
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
        format=Config.LOG_FORMAT
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ==========================
# VALIDATION (SIMPLIFIED)
# ==========================
def validate_glucose_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight validation for Streamlit inputs
    (NOT Flask request validation)
    """

    if not data:
        raise ValueError("Input data is empty")

    try:
        current_glucose = float(data["current_glucose"])
    except Exception:
        raise ValueError("current_glucose must be a number")

    if not (20 <= current_glucose <= 600):
        raise ValueError("Glucose out of range (20–600 mg/dL)")

    predictions = data.get("predictions", [120, 130, 140])
    uncertainty = data.get("uncertainty", {})
    risk = data.get("risk", {"type": "NORMAL"})

    carbs_limit = int(data.get("carbs_limit", Config.DEFAULT_CARBS_LIMIT))
    meal_type = data.get("meal_type", "regular")

    return {
        "current_glucose": current_glucose,
        "predictions": predictions,
        "uncertainty": uncertainty,
        "risk": risk,
        "carbs_limit": carbs_limit,
        "meal_type": meal_type
    }
