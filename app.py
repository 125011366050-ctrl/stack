import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from engine import GlucoseEngine

@st.cache_resource
def load_engine():
    return GlucoseEngine()

engine = load_engine()

st.title("🩺 CDSS - Diabetes Decision Support System")

st.subheader("📥 Enter Last 10 CGM Readings (5-min intervals)")

default_rows = pd.DataFrame({
    "Time": [(datetime.now() - timedelta(minutes=5 * i)).strftime("%H:%M") for i in range(9, -1, -1)],
    "Glucose": [120] * 10,
    "HR": [75] * 10,
    "Meal_Flag": [0] * 10,
    "Carbs": [0] * 10,
    "Protein": [0] * 10,
    "Fat": [0] * 10,
    "Fiber": [0] * 10,
    "Calories": [0] * 10,
})

entries = st.data_editor(
    default_rows,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Glucose": st.column_config.NumberColumn(min_value=40, max_value=400),
        "HR": st.column_config.NumberColumn(min_value=30, max_value=220),
        "Meal_Flag": st.column_config.NumberColumn(min_value=0, max_value=1, step=1),
    }
)
st.caption("Only 10 readings needed — the model pads its 36-step window automatically.")

col1, col2 = st.columns(2)
with col1:
    carbs_limit = st.number_input("Carbs Limit (g)", 0, 200, 50)
with col2:
    meal_type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner"])

if st.button("Run CDSS Analysis"):
    try:
        result = engine.run(
            entries=entries,
            carbs_limit=carbs_limit,
            meal_type=meal_type
        )

        st.subheader("📈 Glucose Forecast")
        preds = result["predictions"]
        c1, c2, c3 = st.columns(3)
        c1.metric("30 min", f"{preds.get('30min', 0)} mg/dL")
        c2.metric("60 min", f"{preds.get('60min', 0)} mg/dL")
        c3.metric("120 min", f"{preds.get('120min', 0)} mg/dL")

        st.subheader("🍽️ Food Recommendations")
        st.dataframe(result["recommendation"]["foods"])

        st.subheader("🏃 Activity Recommendation")
        current_glucose = float(entries["Glucose"].iloc[-1])
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
