import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class FoodRecommender:
    """
    STRICT Excel-based food recommender for CDSS
    PUBLICATION-READY: Bidirectional, phase-aware, with counterfactuals
    """
    
    def __init__(
        self, 
        excel_path: str = "Indian_Foods_GI_GL_Database.xlsx",
        sensitivity: float = 1.0
    ):
        self.sensitivity = sensitivity
        
        try:
            self.df = pd.read_excel(excel_path)
            self.df.columns = [c.strip() for c in self.df.columns]
            
            # Required columns validation
            required_cols = ["Food Name", "GI", "Carbs (g)"]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                raise ValueError(f"❌ Excel missing required columns: {missing_cols}")
            
            # Ensure numeric columns
            self.df["GI"] = pd.to_numeric(self.df["GI"], errors="coerce")
            self.df["Carbs (g)"] = pd.to_numeric(self.df["Carbs (g)"], errors="coerce")
            self.df = self.df.dropna(subset=["GI", "Carbs (g)"])
            
            # GL with safety
            if "GL" not in self.df.columns:
                self.df["GL"] = self.df["GI"] * self.df["Carbs (g)"] / 100
            else:
                self.df["GL"] = pd.to_numeric(self.df["GL"], errors="coerce").fillna(0)
            
            # Ensure all columns exist
            for col in ["Protein (g)", "Fat (g)", "Calories (kcal)"]:
                if col not in self.df.columns:
                    self.df[col] = 0
                else:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
            
            # Normalized nutrition score
            gi_norm = self.df["GI"] / 100
            carb_norm = self.df["Carbs (g)"] / 100
            protein_norm = self.df["Protein (g)"] / 30
            fat_norm = self.df["Fat (g)"] / 20
            
            self.df["nutrition_score"] = (
                -gi_norm * 0.4 +
                protein_norm * 0.3 -
                carb_norm * 0.3 -
                fat_norm * 0.05
            )
            
            # Food classification
            self.df["food_class"] = self._classify_foods(self.df["GI"])
            
            # ✅ FIX: Use these flags explicitly
            self.df["fast_absorbing"] = (
                (self.df["GI"] >= 70) | 
                (self.df["GL"] >= 20) |
                (self.df["food_class"] == "high_gi")
            )
            
            self.df["slow_absorbing"] = (
                (self.df["GI"] <= 55) & 
                (self.df["GL"] <= 15) &
                (self.df["food_class"] == "low_gi") &
                (self.df["Protein (g)"] >= 3)
            )
            
            logger.info(f"✅ Loaded {len(self.df)} foods from Excel")
            logger.info(f"Fast-absorbing: {self.df['fast_absorbing'].sum()}")
            logger.info(f"Slow-absorbing: {self.df['slow_absorbing'].sum()}")
            
        except FileNotFoundError:
            logger.error(f"❌ Excel file not found: {excel_path}")
            raise FileNotFoundError(f"Required Excel file not found: {excel_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load Excel: {e}")
            raise ValueError(f"Failed to load Excel: {e}")
    
    def _classify_foods(self, gi_series: pd.Series) -> np.ndarray:
        """Vectorized classification"""
        return np.where(
            gi_series >= 70, "high_gi",
            np.where(gi_series >= 55, "medium_gi", "low_gi")
        )
    
    def _get_phase_specific_foods(
        self, 
        df: pd.DataFrame, 
        phase: str,
        severity: str = "moderate"
    ) -> Tuple[pd.DataFrame, float, str, str]:
        """
        Get foods specific to glucose phase with standardized return
        """
        if phase == "rising":
            # ✅ FIX: Use slow_absorbing flag
            food_pool = df[df["slow_absorbing"] == True]
            
            if severity == "severe":
                food_pool = food_pool[
                    (food_pool["GI"] <= 45) & 
                    (food_pool["Protein (g)"] >= 5)
                ]
                max_carbs_multiplier = 0.3
            else:
                food_pool = food_pool[
                    (food_pool["GI"] <= 55) & 
                    (food_pool["Protein (g)"] >= 3)
                ]
                max_carbs_multiplier = 0.5
            
            if len(food_pool) > 0:
                food_pool = food_pool.copy()
                food_pool["phase_score"] = (
                    -food_pool["GI"] * 0.5 +
                    food_pool["Protein (g)"] * 0.4 -
                    food_pool["Carbs (g)"] * 0.1
                )
            
            strategy = "RISE PREVENTION - GLUCOSE SPIKE CONTROL"
            message = "⚠️ Glucose rising. Choose slow-absorbing foods."
            
        elif phase == "falling":
            # ✅ FIX: Use fast_absorbing flag
            food_pool = df[df["fast_absorbing"] == True]
            
            if severity == "severe":
                food_pool = food_pool[
                    (food_pool["GI"] >= 70) | 
                    (food_pool["GL"] >= 20)
                ]
                max_carbs_multiplier = 1.5
            else:
                food_pool = food_pool[
                    (food_pool["GI"] >= 60) | 
                    (food_pool["GL"] >= 15)
                ]
                max_carbs_multiplier = 1.2
            
            if len(food_pool) > 0:
                food_pool = food_pool.copy()
                food_pool["phase_score"] = (
                    food_pool["GL"] * 0.4 +
                    food_pool["Carbs (g)"] * 0.3 -
                    food_pool["GI"] * 0.1
                )
            
            strategy = "FALL PREVENTION - GLUCOSE STABILIZATION"
            message = "⚠️ Glucose falling. Choose fast-absorbing foods."
            
        elif phase == "peak":
            food_pool = df[
                (df["GI"] >= 45) & (df["GI"] <= 65) &
                (df["GL"] <= 20) &
                (df["Protein (g)"] >= 2)
            ]
            
            if len(food_pool) > 0:
                food_pool = food_pool.copy()
                food_pool["phase_score"] = (
                    -food_pool["GI"] * 0.3 +
                    food_pool["Protein (g)"] * 0.3 -
                    food_pool["Carbs (g)"] * 0.2 +
                    food_pool["Fat (g)"] * 0.1
                )
            
            max_carbs_multiplier = 0.8
            strategy = "PEAK MANAGEMENT - BALANCED GLUCOSE RESPONSE"
            message = "⚠️ Glucose at peak. Choose stabilizing foods."
        
        else:  # normal
            food_pool = df[
                (df["GI"] <= 60) & 
                (df["food_class"].isin(["low_gi", "medium_gi"]))
            ]
            
            if len(food_pool) > 0:
                food_pool = food_pool.copy()
                food_pool["phase_score"] = food_pool["nutrition_score"]
            
            max_carbs_multiplier = 1.0
            strategy = "MAINTENANCE - BALANCED NUTRITION"
            message = "✅ Glucose stable. Maintain balanced eating."
        
        return food_pool, max_carbs_multiplier, strategy, message
    
    def _calculate_counterfactual_impact(
        self,
        selected_food: Dict,
        rejected_food: Dict,
        current_glucose: float
    ) -> Dict:
        """
        🔥 COUNTERFACTUAL EXPLANATION: Quantify impact of rejection
        """
        selected_gi = selected_food.get("GI", 50)
        rejected_gi = rejected_food.get("GI", 50)
        selected_carbs = selected_food.get("Carbs (g)", 0)
        rejected_carbs = rejected_food.get("Carbs (g)", 0)
        
        # Estimate glucose impact difference
        gi_diff = selected_gi - rejected_gi
        carb_diff = selected_carbs - rejected_carbs
        
        # Simplified glucose impact estimation
        estimated_impact = (gi_diff * 0.5 + carb_diff * 0.3)
        
        return {
            "rejected_food": rejected_food.get("Food Name", "Unknown"),
            "gi_difference": float(gi_diff),
            "carb_difference": float(carb_diff),
            "estimated_glucose_impact": float(estimated_impact),
            "reason": f"Rejected {rejected_food.get('Food Name', 'Unknown')} due to {abs(gi_diff):.0f} point GI difference"
        }
    
    def _generate_explanation(
        self,
        food_item: Dict,
        phase: str,
        glucose_context: Dict,
        reason: str,
        counterfactual: Optional[Dict] = None
    ) -> Dict:
        """
        Enhanced explanation with counterfactual reasoning
        """
        explanation = {
            "food_name": food_item.get("Food Name", "Unknown"),
            "primary_reason": reason,
            "phase": phase,
            "clinical_factors": [],
            "ranking_factors": [],
            "counterfactual": counterfactual if counterfactual else None
        }
        
        gi = food_item.get("GI", 50)
        carbs = food_item.get("Carbs (g)", 0)
        protein = food_item.get("Protein (g)", 0)
        gl = food_item.get("GL", 0)
        
        # Phase-specific reasoning
        if phase == "rising":
            if gi <= 55:
                explanation["clinical_factors"].append(
                    f"Low GI ({gi}) slows glucose absorption"
                )
            if protein >= 5:
                explanation["clinical_factors"].append(
                    f"Protein ({protein}g) helps stabilize glucose"
                )
            if carbs <= 15:
                explanation["clinical_factors"].append(
                    f"Low carbs ({carbs}g) prevents additional spike"
                )
                
        elif phase == "falling":
            if gi >= 70:
                explanation["clinical_factors"].append(
                    f"High GI ({gi}) provides rapid glucose rise"
                )
            if gl >= 15:
                explanation["clinical_factors"].append(
                    f"Moderate GL ({gl}) for sustained response"
                )
            if carbs >= 20:
                explanation["clinical_factors"].append(
                    f"Higher carbs ({carbs}g) for glucose recovery"
                )
                
        elif phase == "peak":
            if 45 <= gi <= 65:
                explanation["clinical_factors"].append(
                    f"Medium GI ({gi}) for balanced response"
                )
            if protein >= 3:
                explanation["clinical_factors"].append(
                    f"Protein ({protein}g) helps reduce peak"
                )
        
        # Ranking justification
        phase_score = food_item.get("phase_score", food_item.get("nutrition_score", 0))
        explanation["ranking_factors"].append(
            f"Phase score: {phase_score:.2f}"
        )
        
        return explanation
    
    def _get_safe_fallback(self, df: pd.DataFrame, max_carbs: float, phase: str) -> pd.DataFrame:
        """Safe fallback respecting phase"""
        if phase == "falling":
            return df[
                (df["GI"] >= 50) &
                (df["Carbs (g)"] <= max_carbs * 1.5)
            ]
        elif phase == "rising":
            return df[
                (df["GI"] <= 65) &
                (df["Carbs (g)"] <= max_carbs * 1.2)
            ]
        else:
            return df[
                (df["GI"] <= 70) &
                (df["Carbs (g)"] <= max_carbs * 1.3)
            ]
    
    def _get_phase_priority(self, phase: str) -> int:
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
        return priorities.get(phase, 1)
    
    def recommend(
        self,
        risk: Dict[str, Any],
        predictions: Dict[str, float],
        uncertainty: Dict[str, Dict[str, float]],
        current_glucose: float,
        carbs_limit: int = 30,
        meal_type: str = "regular"
    ) -> Dict[str, Any]:
        """
        BIDIRECTIONAL food recommendation with counterfactual reasoning
        """
        # Validate
        if uncertainty is None or not uncertainty:
            raise ValueError("❌ Uncertainty bounds required")
        
        # Get risk parameters
        risk_type = risk.get("type", "NORMAL")
        trend = risk.get("trend", "STABLE")
        trend_slope = risk.get("trend_slope", 0)
        risk_level = risk.get("level", "LOW")
        
        df = self.df.copy()
        
        # ==============================
        # EXTRACT PREDICTIONS
        # ==============================
        p30 = predictions.get("30min", current_glucose)
        p60 = predictions.get("60min", current_glucose)
        p120 = predictions.get("120min", current_glucose)
        
        lower_30 = uncertainty.get("30min", {}).get("lower", p30 - 10)
        upper_30 = uncertainty.get("30min", {}).get("upper", p30 + 10)
        lower_60 = uncertainty.get("60min", {}).get("lower", p60 - 10)
        upper_60 = uncertainty.get("60min", {}).get("upper", p60 + 10)
        lower_120 = uncertainty.get("120min", {}).get("lower", p120 - 10)
        upper_120 = uncertainty.get("120min", {}).get("upper", p120 + 10)
        
        predicted_peak = 0.5 * upper_30 + 0.3 * upper_60 + 0.2 * upper_120
        predicted_trough = 0.5 * lower_30 + 0.3 * lower_60 + 0.2 * lower_120
        
        raw_peak = max(upper_30, upper_60, upper_120)
        raw_trough = min(lower_30, lower_60, lower_120)
        
        # ==============================
        # ✅ FIX: DIRECTIONAL RISK SCORE
        # ==============================
        hypo_risk = max(0, 70 - predicted_trough)
        hyper_risk = max(0, predicted_peak - 180)
        
        # Apply direction weighting
        is_rising = trend == "RISING"
        is_falling = trend == "FALLING"
        
        base_risk_score = (
            hyper_risk * (1.0 if is_rising else 0.7) +
            hypo_risk * (1.0 if is_falling else 0.7)
        ) / 50  # Normalize
        
        risk_score = base_risk_score * self.sensitivity
        
        # Action level
        if risk_score > 2.0:
            action_level = "CRITICAL"
        elif risk_score > 1.0:
            action_level = "HIGH"
        elif risk_score > 0.5:
            action_level = "MEDIUM"
        else:
            action_level = "LOW"
        
        # ==============================
        # ✅ FIX: PHASE DETECTION WITH PRIORITY
        # ==============================
        
        # Define phases with priority
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
        if current_glucose > 140 and abs(trend_slope) <= 2:
            phases.append(("peak", "peak", "moderate"))
        
        # Borderline
        if (70 <= raw_trough <= 80) or (160 <= raw_peak <= 180):
            phases.append(("borderline", "normal", "moderate"))
        
        # Normal (fallback)
        if not phases:
            phases.append(("stable", "normal", "moderate"))
        
        # ✅ FIX: Select highest priority phase
        selected_phase = max(phases, key=lambda x: self._get_phase_priority(x[0]))
        phase_id, phase, severity = selected_phase
        
        # ==============================
        # GET PHASE-SPECIFIC FOODS
        # ==============================
        food_pool, carb_mult, strategy, message = self._get_phase_specific_foods(
            df, phase, severity
        )
        max_carbs = carbs_limit * carb_mult
        
        # Apply filters
        if "phase_score" in food_pool.columns:
            food_pool = food_pool.sort_values("phase_score", ascending=False)
        else:
            food_pool = food_pool.sort_values("nutrition_score", ascending=False)
        
        food_pool = food_pool[food_pool["Carbs (g)"] <= max_carbs]
        
        # Safe fallback
        if food_pool.empty:
            logger.warning(f"⚠️ No foods matched, using safe fallback for phase: {phase}")
            food_pool = self._get_safe_fallback(df, max_carbs, phase)
            food_pool = food_pool.sort_values("nutrition_score", ascending=False)
        
        food_pool = food_pool.drop_duplicates(subset=["Food Name"])
        top_foods = food_pool.head(8)
        
        # ==============================
        # 🔥 COUNTERFACTUAL REASONING
        # ==============================
        # Get rejected alternatives for counterfactual
        all_foods = df.head(20)
        rejected_foods = all_foods[
            ~all_foods["Food Name"].isin(top_foods["Food Name"].tolist())
        ].head(3)
        
        # Generate counterfactuals for top foods
        foods_with_explanations = []
        for idx, (_, row) in enumerate(top_foods.iterrows()):
            food_dict = row.to_dict()
            
            # Get a rejected food for comparison
            counterfactual = None
            if len(rejected_foods) > idx:
                rejected_row = rejected_foods.iloc[idx]
                counterfactual = self._calculate_counterfactual_impact(
                    food_dict,
                    rejected_row.to_dict(),
                    current_glucose
                )
            
            # Generate explanation
            reason = f"Selected for {phase} phase management"
            explanation = self._generate_explanation(
                food_dict,
                phase,
                {
                    "current": current_glucose,
                    "predicted_peak": float(predicted_peak),
                    "predicted_trough": float(predicted_trough),
                    "trend": trend,
                    "trend_slope": float(trend_slope)
                },
                reason,
                counterfactual
            )
            food_dict["explanation"] = explanation
            foods_with_explanations.append(food_dict)
        
        # ==============================
        # GLUCOSE CONTEXT
        # ==============================
        glucose_context = {
            "current": current_glucose,
            "predicted_peak": float(predicted_peak),
            "predicted_trough": float(predicted_trough),
            "raw_peak_upper": float(raw_peak),
            "raw_trough_lower": float(raw_trough),
            "trend": trend,
            "trend_slope": float(trend_slope),
            "phase": phase,
            "phase_id": phase_id,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "hypo_risk": round(hypo_risk, 2),
            "hyper_risk": round(hyper_risk, 2)
        }
        
        # ==============================
        # CLINICAL STATE MAPPING
        # ==============================
        clinical_state_map = {
            "severe_hypo": "SEVERE_HYPOGLYCEMIA",
            "hypo": "HYPOGLYCEMIA",
            "falling_fast": "FALLING_FAST",
            "falling_moderate": "FALLING_MODERATE",
            "severe_hyper": "SEVERE_HYPERGLYCEMIA",
            "hyper": "HYPERGLYCEMIA",
            "rising_fast": "RISING_FAST",
            "rising_moderate": "RISING_MODERATE",
            "peak": "PEAK_GLUCOSE",
            "borderline": "BORDERLINE",
            "stable": "NORMAL"
        }
        clinical_state = clinical_state_map.get(phase_id, "NORMAL")
        
        # ==============================
        # RESPONSE
        # ==============================
        return {
            "strategy": strategy,
            "phase": phase,
            "phase_id": phase_id,
            "urgency": "CRITICAL" if action_level == "CRITICAL" else "IMMEDIATE" if action_level == "HIGH" else "SOON" if action_level == "MEDIUM" else "NORMAL",
            "action_level": action_level,
            "clinical_action_score": round(risk_score, 2),
            "base_risk_score": round(base_risk_score, 2),
            "risk_multiplier": self.sensitivity,
            "message": message,
            "meal_timing": "NOW" if action_level == "CRITICAL" else "Within 5 minutes" if action_level == "HIGH" else "Within 30 minutes" if action_level == "MEDIUM" else "Next scheduled meal",
            "clinical_reason": reason,
            "clinical_state": clinical_state,
            "personalization_sensitivity": self.sensitivity,
            "glucose_context": glucose_context,
            "filters_applied": {
                "max_carbs": float(max_carbs),
                "phase": phase,
                "gi_threshold": "Low GI" if phase == "rising" else "High GI" if phase == "falling" else "Balanced"
            },
            "foods": foods_with_explanations,
            "total_recommendations": len(foods_with_explanations)
        }
    
    def set_sensitivity(self, sensitivity: float):
        if sensitivity <= 0 or sensitivity > 3.0:
            raise ValueError("Sensitivity must be between 0 and 3.0")
        self.sensitivity = sensitivity
        logger.info(f"✅ Updated sensitivity to: {sensitivity}")
    
    def get_food_by_gi_band(self, gi_band: str) -> List[Dict]:
        if gi_band.lower() == "low":
            foods = self.df[self.df["GI"] <= 55].head(20).to_dict(orient="records")
        elif gi_band.lower() == "medium":
            foods = self.df[(self.df["GI"] > 55) & (self.df["GI"] <= 70)].head(20).to_dict(orient="records")
        elif gi_band.lower() == "high":
            foods = self.df[self.df["GI"] > 70].head(20).to_dict(orient="records")
        else:
            foods = []
        return foods
    
    def search_food(self, query: str) -> List[Dict]:
        if query:
            foods = self.df[
                self.df["Food Name"].str.contains(query, case=False)
            ].head(20).to_dict(orient="records")
            return foods
        return []

    def get_all_foods(self, limit: int = 50) -> List[Dict]:
        return self.df.head(limit).to_dict(orient="records")
