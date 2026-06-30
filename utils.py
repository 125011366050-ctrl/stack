import logging
from typing import Dict, Any
from config import Config

def setup_logging():
    """Configure application-wide logging"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
        format=Config.LOG_FORMAT
    )
    return logging.getLogger(__name__)


def validate_glucose_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate incoming request payload for the recommend endpoint.
    Raises ValueError on invalid input.
    """
    if not data:
        raise ValueError("Request body is empty")

    if "current_glucose" not in data:
        raise ValueError("Missing required field: current_glucose")

    try:
        current_glucose = float(data["current_glucose"])
    except (TypeError, ValueError):
        raise ValueError("current_glucose must be a number")

    if current_glucose < 20 or current_glucose > 600:
        raise ValueError("current_glucose out of plausible physiological range (20-600 mg/dL)")

    predictions = data.get("predictions", {})
    uncertainty = data.get("uncertainty", {})
    risk = data.get("risk", {"type": "NORMAL", "trend": "STABLE", "trend_slope": 0})

    if not isinstance(predictions, dict):
        raise ValueError("predictions must be an object")
    if not isinstance(uncertainty, dict) or not uncertainty:
        raise ValueError("uncertainty bounds are required for clinical safety")

    carbs_limit = data.get("carbs_limit", Config.DEFAULT_CARBS_LIMIT)
    try:
        carbs_limit = int(carbs_limit)
    except (TypeError, ValueError):
        raise ValueError("carbs_limit must be an integer")

    meal_type = data.get("meal_type", "regular")

    return {
        "current_glucose": current_glucose,
        "predictions": predictions,
        "uncertainty": uncertainty,
        "risk": risk,
        "carbs_limit": carbs_limit,
        "meal_type": meal_type
    }


def error_response(message: str, status_code: int = 400) -> tuple:
    """Standard error response shape"""
    return {"error": True, "message": message}, status_code


def success_response(payload: Dict[str, Any], status_code: int = 200) -> tuple:
    """Standard success response shape"""
    return {"error": False, "data": payload}, status_code
