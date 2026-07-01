import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class FoodRecommender:
    def __init__(
        self,
        excel_path: str,
        sensitivity: float = 1.0
    ):
        self.sensitivity = sensitivity
        self.df = self._load_data(excel_path)

    # ==========================
    # DATA LOADING (OPTIMIZED)
    # ==========================
    def _load_data(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()

            required = ["Food Name", "GI", "Carbs (g)"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            df["GI"] = pd.to_numeric(df["GI"], errors="coerce")
            df["Carbs (g)"] = pd.to_numeric(df["Carbs (g)"], errors="coerce")

            df = df.dropna(subset=["GI", "Carbs (g)"])

            # GL
            df["GL"] = df["GL"] if "GL" in df.columns else (df["GI"] * df["Carbs (g)"] / 100)

            # optional nutrients
            for c in ["Protein (g)", "Fat (g)", "Calories (kcal)"]:
                if c not in df.columns:
                    df[c] = 0
                else:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            # SIMPLE SCORE (faster than multiple ops)
            df["score"] = (
                -df["GI"] * 0.4 +
                df["Protein (g)"] * 0.3 -
                df["Carbs (g)"] * 0.3 -
                df["Fat (g)"] * 0.05
            )

            logger.info(f"Loaded {len(df)} foods")
            return df

        except Exception as e:
            logger.error(f"Food loading failed: {e}")
            raise

    # ==========================
    # MAIN RECOMMENDER
    # ==========================
    def recommend(
        self,
        risk: Dict[str, Any],
        predictions: Dict[str, float],
        uncertainty: Dict[str, Any],
        current_glucose: float,
        carbs_limit: int = 30,
        meal_type: str = "regular"
    ) -> Dict[str, Any]:

        df = self.df

        # 🔥 dynamic filtering based on risk
        if risk["type"] == "HYPOGLYCEMIA":
            food_pool = df[df["Carbs (g)"] > 10]  # energy foods
            strategy = "LOW GLUCOSE RECOVERY"
        elif risk["type"] == "HYPERGLYCEMIA":
            food_pool = df[df["GI"] <= 50]
            strategy = "LOW GI CONTROL"
        else:
            food_pool = df[df["GI"] <= 60]
            strategy = "BALANCED CONTROL"

        # apply carb limit
        food_pool = food_pool[food_pool["Carbs (g)"] <= carbs_limit]

        # rank
        food_pool = food_pool.sort_values("score", ascending=False)

        top = food_pool.head(8)

        return {
            "strategy": strategy,
            "message": "Personalized CDSS recommendation",
            "foods": top[
                ["Food Name", "GI", "GL", "Carbs (g)",
                 "Protein (g)", "Fat (g)", "Calories (kcal)", "score"]
            ].to_dict(orient="records"),
            "count": len(top)
        }

    # ==========================
    # UTILITIES (FAST)
    # ==========================
    def set_sensitivity(self, s: float):
        if not 0 < s <= 3:
            raise ValueError("Sensitivity must be 0–3")
        self.sensitivity = s

    def get_food_by_gi_band(self, band: str) -> List[Dict]:
        if band == "low":
            df = self.df[self.df["GI"] <= 55]
        elif band == "medium":
            df = self.df[(self.df["GI"] > 55) & (self.df["GI"] <= 70)]
        else:
            df = self.df[self.df["GI"] > 70]

        return df.head(20).to_dict("records")

    def search_food(self, query: str) -> List[Dict]:
        if not query:
            return []
        return self.df[
            self.df["Food Name"].str.contains(query, case=False, na=False)
        ].head(20).to_dict("records")

    def get_all_foods(self, limit: int = 50) -> List[Dict]:
        return self.df.head(limit).to_dict("records")
