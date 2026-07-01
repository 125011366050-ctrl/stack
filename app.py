import streamlit as st
from engine import GlucoseEngine
from utils import validate_glucose_input

@st.cache_resource
def load_engine():
    return GlucoseEngine()

engine = load_engine()

st.title("🩺 CDSS - Diabetes Decision Support System")

col1, col2 = st.columns(2)
with col1:
    risk = st.selectbox("Risk Level", ["low", "medium", "high"])
    current_glucose = st.number_input("Current Glucose (mg/dL)", 40, 400, 120)
    carbs_limit = st.number_input("Carbs Limit (g)", 0, 200, 50)
with col2:
    meal_type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner"])
    predictions = st.text_input("Past Predictions (comma-separated)", "120,130,140")
    uncertainty = st.number_input("Uncertainty", 0.0, 1.0, 0.2)

if st.button("Run CDSS Analysis"):
    try:
        history = [float(x) for x in predictions.split(",")] + [current_glucose]

        payload = validate_glucose_input({
            "current_glucose": current_glucose,
            "carbs_limit": carbs_limit,
            "meal_type": meal_type,
            "predictions": history,
            "uncertainty": uncertainty,
            "risk": {"type": "NORMAL"}
        })

        result = engine.run(
            glucose_history=history,
            carbs_limit=payload["carbs_limit"],
            meal_type=payload["meal_type"]
        )

        st.subheader("📈 Glucose Forecast")
        preds = result["predictions"]
        c1, c2, c3 = st.columns(3)
        c1.metric("30 min", f"{preds.get('30min', 0)} mg/dL")
        c2.metric("60 min", f"{preds.get('60min', 0)} mg/dL")
        c3.metric("120 min", f"{preds.get('120min', 0)} mg/dL")

        st.subheader("🍽️ Food Recommendations")
        foods = result["recommendation"]["foods"]
        st.dataframe(foods)

        st.subheader("🏃 Activity Recommendation")
        if current_glucose > 180:
            activity = "🚶 Light walking (10–15 min) recommended"
        elif current_glucose < 80:
            activity = "🍯 Consume quick glucose + rest"
        else:
            activity = "🏃 Moderate activity (20–30 min walk or cycling)"
        st.info(activity)

        st.subheader("🧠 Clinical Strategy")
        st.success(result["recommendation"]["strategy"])

    except Exception as e:
        st.error(str(e))
