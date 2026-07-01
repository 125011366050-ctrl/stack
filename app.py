import streamlit as st
from recommender import FoodRecommender
from config import Config
from utils import validate_glucose_payload

@st.cache_resource
def load_model():
    return FoodRecommender(
        excel_path=Config.EXCEL_PATH,
        sensitivity=Config.DEFAULT_SENSITIVITY
    )

recommender = load_model()

st.title("🍎 CDSS Food Recommender")

# Inputs
risk = st.selectbox("Risk Level", ["low", "medium", "high"])
current_glucose = st.number_input("Current Glucose", 40, 400, 120)
carbs_limit = st.number_input("Carbs Limit", 0, 200, 50)
meal_type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner"])

predictions = st.text_input("Predictions (comma-separated)", "120,130,140")
uncertainty = st.number_input("Uncertainty", 0.0, 1.0, 0.2)

if st.button("Get Recommendation"):
    try:
        payload = validate_glucose_payload({
            "risk": risk,
            "current_glucose": current_glucose,
            "carbs_limit": carbs_limit,
            "meal_type": meal_type,
            "predictions": [float(x) for x in predictions.split(",")],
            "uncertainty": uncertainty
        })

        result = recommender.recommend(
            risk=payload["risk"],
            predictions=payload["predictions"],
            uncertainty=payload["uncertainty"],
            current_glucose=payload["current_glucose"],
            carbs_limit=payload["carbs_limit"],
            meal_type=payload["meal_type"]
        )

        st.success("Recommendation generated")
        st.json(result)

    except Exception as e:
        st.error(str(e))
