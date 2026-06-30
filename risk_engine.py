"""
RISK ENGINE - Priority-based phase detection
"""
from typing import Dict, Any, List, Tuple

def get_phase_priority(phase_id: str) -> int:
    """Priority scoring for phase selection"""
    priorities = {
        "severe_hypo": 6,
        "falling_fast": 5,
        "severe_hyper": 5,
        "falling_moderate": 4,
        "rising_fast": 4,
        "rising_moderate": 3,
        "peak": 2,
        "borderline": 2,
        "stable": 1
    }
    return priorities.get(phase_id, 1)

def detect_phases(
    raw_trough: float,
    raw_peak: float,
    trend_slope: float,
    trend: str,
    current: float
) -> List[Tuple[str, str, str]]:
    """
    Detect all possible phases with priority
    Returns: List of (phase_id, phase, severity)
    """
    phases = []
    
    # Severe hypoglycemia (highest priority)
    if raw_trough < 54:
        phases.append(("severe_hypo", "falling", "severe"))
    elif raw_trough < 70:
        phases.append(("hypo", "falling", "moderate"))
    
    # Severe hyperglycemia
    if raw_peak > 300:
        phases.append(("severe_hyper", "rising", "severe"))
    elif raw_peak > 180:
        phases.append(("hyper", "rising", "moderate"))
    
    # Trend-based phases
    if trend == "FALLING" and trend_slope < -5:
        phases.append(("falling_fast", "falling", "moderate"))
    elif trend == "FALLING" and -5 <= trend_slope <= -2:
        phases.append(("falling_moderate", "falling", "moderate"))
    
    if trend == "RISING" and trend_slope > 5:
        phases.append(("rising_fast", "rising", "moderate"))
    elif trend == "RISING" and 2 <= trend_slope <= 5:
        phases.append(("rising_moderate", "rising", "moderate"))
    
    # Peak detection
    if current > 140 and abs(trend_slope) <= 2:
        phases.append(("peak", "peak", "moderate"))
    
    # Borderline
    if (70 <= raw_trough <= 80) or (160 <= raw_peak <= 180):
        phases.append(("borderline", "normal", "moderate"))
    
    # Normal (fallback)
    if not phases:
        phases.append(("stable", "normal", "moderate"))
    
    return phases

def classify_risk_and_phase(
    raw_trough: float,
    raw_peak: float,
    trend_slope: float,
    trend: str,
    current: float
) -> Dict[str, Any]:
    """
    Classify risk AND detect glucose phase with priority
    """
    # Detect all phases
    phases = detect_phases(raw_trough, raw_peak, trend_slope, trend, current)
    
    # Select highest priority phase
    selected = max(phases, key=lambda x: get_phase_priority(x[0]))
    phase_id, phase, severity = selected
    
    # ==============================
    # RISK CLASSIFICATION
    # ==============================
    # Priority 1: Severe Hypoglycemia
    if raw_trough < 54:
        return {
            "level": "CRITICAL",
            "type": "SEVERE_HYPOGLYCEMIA",
            "phase": phase,
            "phase_id": phase_id,
            "trend": trend,
            "trend_slope": float(trend_slope),
            "severity_score": float(54 - raw_trough),
            "risk_probability": 0.95,
            "action_required": "IMMEDIATE"
        }
    
    # Priority 2: Hypoglycemia
    elif raw_trough < 70:
        return {
            "level": "HIGH",
            "type": "HYPOGLYCEMIA",
            "phase": phase,
            "phase_id": phase_id,
            "trend": trend,
            "trend_slope": float(trend_slope),
            "severity_score": float(70 - raw_trough),
            "risk_probability": 0.80,
            "action_required": "URGENT"
        }
    
    # Priority 3: Severe Hyperglycemia
    elif raw_peak > 300:
        return {
            "level": "CRITICAL",
            "type": "SEVERE_HYPERGLYCEMIA",
            "phase": phase,
            "phase_id": phase_id,
            "trend": trend,
            "trend_slope": float(trend_slope),
            "severity_score": float(raw_peak - 300),
            "risk_probability": 0.95,
            "action_required": "IMMEDIATE"
        }
    
    # Priority 4: Hyperglycemia
    elif raw_peak > 180:
        return {
            "level": "HIGH",
            "type": "HYPERGLYCEMIA",
            "phase": phase,
            "phase_id": phase_id,
            "trend": trend,
            "trend_slope": float(trend_slope),
            "severity_score": float(raw_peak - 180),
            "risk_probability": 0.80,
            "action_required": "URGENT"
        }
    
    # Priority 5: Phase-based alerts
    elif phase_id in ["falling_fast", "falling_moderate", "rising_fast", "rising_moderate"]:
        return {
            "level": "MEDIUM",
            "type": phase_id.upper(),
            "phase": phase,
            "phase_id": phase_id,
            "trend": trend,
            "trend_slope": float(trend_slope),
            "severity_score": float(abs(trend_slope)),
            "risk_probability": 0.60,
            "action_required": "MONITOR"
        }
    
    # Priority 6: Peak
    elif phase_id == "peak":
        return {
            "level": "MEDIUM",
            "type": "PEAK_GLUCOSE",
            "phase": phase,
            "phase_id": phase_id,
            "trend": trend,
            "trend_slope": float(trend_slope),
            "severity_score": float(max(0, current - 140)),
            "risk_probability": 0.55,
            "action_required": "MONITOR"
        }
    
    # Priority 7: Borderline
    elif phase_id == "borderline":
        return {
            "level": "MEDIUM",
            "type": "BORDERLINE",
            "phase": phase,
            "phase_id": phase_id,
            "trend": trend,
            "trend_slope": float(trend_slope),
            "severity_score": 0.0,
            "risk_probability": 0.50,
            "action_required": "MONITOR"
        }
    
    # Priority 8: Normal
    else:
        return {
            "level": "LOW",
            "type": "NORMAL",
            "phase": phase,
            "phase_id": phase_id,
            "trend": trend,
            "trend_slope": float(trend_slope),
            "severity_score": 0.0,
            "risk_probability": 0.10,
            "action_required": "ROUTINE"
        }
