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
        self.df = self._load_excel(excel_path)

    # ==========================
    # LOAD EXCEL (STRICT)
    # ==========================
    def _load_excel(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()

            required = ["Food Name", "GI", "Carbs (g)"]
            missing = [c for c in required if c not in df.columns]

            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # numeric conversion
            df["GI"] = pd.to_numeric(df["GI"], errors="coerce")
            df["Carbs (g)"] = pd.to_numeric(df["Carbs (g)"], errors="coerce")

            df = df.dropna(subset=["GI", "Carbs (g)"])

            # GL (strict Excel or computed fallback)
            if "GL" not in df.columns:
                df["GL"] = df["GI"] * df["Carbs (g)"] / 100
            else:
                df["GL"] = pd.to_numeric(df["GL"], errors="coerce").fillna(0)

            # optional nutrients
            for col in ["Protein (g)", "Fat (g)", "Calories (kcal)"]:
                if col not in df.columns:
                    df[col] = 0
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # classification
            df["food_class"] = np.where(
                df["GI"] >= 70, "high_gi",
                np.where(df["GI"] >= 55, "medium_gi", "low_gi")
            )

            # absorption flags
            df["fast_absorbing"] = (df["GI"] >= 70) | (df["GL"] >= 20)
            df["slow_absorbing"] = (df["GI"] <= 55) & (df["GL"] <= 15)

            logger.info(f"Loaded {len(df)} foods from Excel")

            return df

        except FileNotFoundError:
            logger.error(f"Excel not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Excel load failed: {e}")
            raise

    # ==========================
    # RECOMMENDATION ENGINE
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

        risk_type = risk.get("type", "NORMAL")

        # --------------------------
        # STRATEGY LOGIC
        # --------------------------
        if risk_type == "HYPOGLYCEMIA":
            pool = df[df["Carbs (g)"] > 10]
            strategy = "HYPO RECOVERY"
        elif risk_type == "HYPERGLYCEMIA":
            pool = df[df["GI"] <= 50]
            strategy = "GLUCOSE CONTROL"
        else:
            pool = df[df["GI"] <= 60]
            strategy = "BALANCED"

        # apply carb constraint
        pool = pool[pool["Carbs (g)"] <= carbs_limit]

        # ranking logic (simple but stable)
        pool = pool.copy()
        pool["rank_score"] = (
            -pool["GI"] * 0.4 +
            pool["Protein (g)"] * 0.3 -
            pool["Carbs (g)"] * 0.3 -
            pool["Fat (g)"] * 0.05
        )

        top = pool.sort_values("rank_score", ascending=False).head(8)

        return {
            "strategy": strategy,
            "message": "Strict Excel-based CDSS recommendation",
            "foods": top[[
                "Food Name", "GI", "GL", "Carbs (g)",
                "Protein (g)", "Fat (g)", "Calories (kcal)",
                "food_class", "rank_score"
            ]].to_dict(orient="records"),
            "count": len(top)
        }

    # ==========================
    # UTILITIES
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
