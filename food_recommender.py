import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class FoodRecommender:
    def __init__(
        self, 
        excel_path: str = "Indian_Foods_GI_GL_Database.xlsx",
        sensitivity: float = 1.0
    ):
        self.sensitivity = sensitivity
        
        try:
            self.df = pd.read_excel(excel_path)
            self.df.columns = [c.strip() for c in self.df.columns]
            
            required_cols = ["Food Name", "GI", "Carbs (g)"]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                raise ValueError(f"❌ Excel missing required columns: {missing_cols}")
            
            self.df["GI"] = pd.to_numeric(self.df["GI"], errors="coerce")
            self.df["Carbs (g)"] = pd.to_numeric(self.df["Carbs (g)"], errors="coerce")
            self.df = self.df.dropna(subset=["GI", "Carbs (g)"])
            
            if "GL" not in self.df.columns:
                self.df["GL"] = self.df["GI"] * self.df["Carbs (g)"] / 100
            else:
                self.df["GL"] = pd.to_numeric(self.df["GL"], errors="coerce").fillna(0)
            
            for col in ["Protein (g)", "Fat (g)", "Calories (kcal)"]:
                if col not in self.df.columns:
                    self.df[col] = 0
                else:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
            
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
            
            self.df["food_class"] = self._classify_foods(self.df["GI"])
            
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
            
        except FileNotFoundError:
            logger.error(f"❌ Excel file not found: {excel_path}")
            raise FileNotFoundError(f"Required Excel file not found: {excel_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load Excel: {e}")
            raise ValueError(f"Failed to load Excel: {e}")
    
    def _classify_foods(self, gi_series: pd.Series) -> np.ndarray:
        return np.where(
            gi_series >= 70, "high_gi",
            np.where(gi_series >= 55, "medium_gi", "low_gi")
        )
    
    def recommend(
        self,
        risk: Dict[str, Any],
        predictions: Dict[str, float],
        uncertainty: Dict[str, Dict[str, float]],
        current_glucose: float,
        carbs_limit: int = 30,
        meal_type: str = "regular"
    ) -> Dict[str, Any]:
        # Simple recommendation for now
        df = self.df.copy()
        food_pool = df[df["GI"] <= 60]
        food_pool = food_pool.sort_values("nutrition_score", ascending=False)
        top_foods = food_pool.head(8)
        
        return {
            "strategy": "MAINTENANCE - BALANCED",
            "phase": "stable",
            "message": "✅ Balanced recommendations",
            "foods": top_foods[[
                "Food Name", "GI", "GL", "Carbs (g)", 
                "Protein (g)", "Fat (g)", "Calories (kcal)",
                "food_class", "nutrition_score"
            ]].to_dict(orient="records"),
            "total_recommendations": len(top_foods)
        }
    
    def set_sensitivity(self, sensitivity: float):
        if sensitivity <= 0 or sensitivity > 3.0:
            raise ValueError("Sensitivity must be between 0 and 3.0")
        self.sensitivity = sensitivity
    
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
