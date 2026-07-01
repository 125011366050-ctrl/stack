import streamlit as st
from recommender import FoodRecommender
from config import Config
from utils import validate_glucose_payload

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return FoodRecommender(
        excel_path=Config.EXCEL_PATH,
        sensitivity=Config.DEFAULT_SENSITIVITY
    )

recommender = load_model()

# ==========================
# UI HEADER
# ==========================
st.title("🩺 CDSS - Diabetes Decision Support System")

# ==========================
# INPUTS
# ==========================
col1, col2 = st.columns(2)

with col1:
    risk = st.selectbox("Risk Level", ["low", "medium", "high"])
    current_glucose = st.number_input("Current Glucose (mg/dL)", 40, 400, 120)
    carbs_limit = st.number_input("Carbs Limit (g)", 0, 200, 50)

with col2:
    meal_type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner"])
    predictions = st.text_input("Past Predictions (comma-separated)", "120,130,140")
    uncertainty = st.number_input("Uncertainty", 0.0, 1.0, 0.2)

# ==========================
# RUN BUTTON
# ==========================
if st.button("Run CDSS Analysis"):

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

        # ==========================
        # 📈 GLUCOSE PREDICTION VIEW
        # ==========================
        st.subheader("📈 Glucose Forecast")

        preds = result.get("predictions", {})
        c1, c2, c3 = st.columns(3)

        c1.metric("30 min", f"{preds.get('30min', 0)} mg/dL")
        c2.metric("60 min", f"{preds.get('60min', 0)} mg/dL")
        c3.metric("120 min", f"{preds.get('120min', 0)} mg/dL")

        # ==========================
        # 🍽️ FOOD RECOMMENDATIONS
        # ==========================
        st.subheader("🍽️ Food Recommendations")

        foods = result["recommendation"]["foods"]
        st.dataframe(foods)

        # ==========================
        # 🏃 ACTIVITY RECOMMENDATION (NEW)
        # ==========================
        st.subheader("🏃 Activity Recommendation")

        glucose = current_glucose

        if glucose > 180:
            activity = "🚶 Light walking (10–15 min) recommended"
        elif glucose < 80:
            activity = "🍯 Consume quick glucose + rest"
        else:
            activity = "🏃 Moderate activity (20–30 min walk or cycling)"

        st.info(activity)

        # ==========================
        # 🧠 STRATEGY
        # ==========================
        st.subheader("🧠 Clinical Strategy")
        st.success(result["recommendation"]["strategy"])

    except Exception as e:
        st.error(str(e))
