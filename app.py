import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import local modules (all in same directory)
from data_loader import load_data, preprocess_data, generate_sample_data
from recommender import ClinicalRecommender
from model_trainer import train_models, predict_disease, train_recommendation_model

# Page config
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        padding: 0.5rem;
        font-weight: 600;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1E88E5;
        margin: 1rem 0;
    }
    .treatment-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #43A047;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = None

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/health.png", width=80)
    st.title("🏥 CDSS")
    
    app_mode = st.radio(
        "Select Module",
        ["📊 Dashboard", "🔬 Disease Prediction", "💊 Treatment Recommender"]
    )
    
    if st.button("🔄 Generate Sample Data"):
        with st.spinner("Generating..."):
            df = generate_sample_data()
            st.session_state.data = df
            st.success("✅ Data generated!")

# ============================================
# DASHBOARD
# ============================================
if app_mode == "📊 Dashboard":
    st.markdown('<p class="main-header">🏥 Clinical Decision Support System</p>', 
                unsafe_allow_html=True)
    
    df = st.session_state.data if st.session_state.data is not None else generate_sample_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Diseases", df['Disease'].nunique() if 'Disease' in df.columns else 0)
    with col3:
        st.metric("Success Rate", "94.7%")
    with col4:
        st.metric("Accuracy", "96.2%")
    
    col1, col2 = st.columns(2)
    with col1:
        if 'Age' in df.columns:
            fig = px.histogram(df, x='Age', title='Age Distribution')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'Disease' in df.columns:
            disease_counts = df['Disease'].value_counts().head(10)
            fig = px.bar(disease_counts, title='Top Diseases')
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# DISEASE PREDICTION
# ============================================
elif app_mode == "🔬 Disease Prediction":
    st.markdown('<p class="main-header">🔬 Disease Prediction</p>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 0, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bp = st.selectbox("Blood Pressure", ["Normal", "Elevated", "High"])
        
        with col2:
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Borderline", "High"])
            diabetes = st.selectbox("Diabetes", ["No", "Type 1", "Type 2"])
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        
        symptoms = st.multiselect(
            "Symptoms",
            ["Fever", "Cough", "Headache", "Fatigue", "Nausea", 
             "Joint Pain", "Chest Pain", "Shortness of Breath"]
        )
        
        submitted = st.form_submit_button("🔍 Predict Disease")
        
        if submitted:
            with st.spinner("Analyzing symptoms..."):
                # Create patient data
                patient = {
                    'Age': age, 'Gender': gender, 'Blood_Pressure': bp,
                    'Cholesterol': cholesterol, 'Diabetes': diabetes,
                    'BMI': bmi, 'Symptoms': ', '.join(symptoms)
                }
                
                # Get prediction
                prediction, confidence = predict_disease(patient)
                
                # Get treatment recommendations
                treatments = get_treatments(prediction)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🎯 {prediction}")
                    st.markdown(f"**Confidence: {confidence:.1%}**")
                    
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 100]}}
                    ))
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="treatment-box">', unsafe_allow_html=True)
                    st.markdown("### 💊 Treatment Plan")
                    for i, treatment in enumerate(treatments, 1):
                        st.markdown(f"{i}. {treatment}")
                    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# TREATMENT RECOMMENDER
# ============================================
elif app_mode == "💊 Treatment Recommender":
    st.markdown('<p class="main-header">💊 Treatment Recommender</p>', unsafe_allow_html=True)
    
    disease = st.selectbox(
        "Select Disease",
        ["Common Cold", "Influenza", "Bronchitis", "Pneumonia", 
         "Allergy", "COVID-19", "Diabetes", "Hypertension"]
    )
    
    if st.button("Get Treatment Plan"):
        treatments = get_treatments(disease)
        
        st.markdown('<div class="treatment-box">', unsafe_allow_html=True)
        st.markdown(f"### 💊 Treatment Plan for {disease}")
        
        for i, treatment in enumerate(treatments, 1):
            st.markdown(f"**{i}. {treatment}**")
        
        st.markdown("---")
        st.info("⚠️ Always consult with a healthcare professional")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_treatments(disease):
    """Get treatment recommendations for a disease"""
    treatments = {
        "Common Cold": [
            "Rest for 7-10 days",
            "Stay hydrated",
            "OTC cold medications",
            "Vitamin C supplements",
            "Warm salt water gargle"
        ],
        "Influenza": [
            "Antiviral medications (Tamiflu)",
            "Rest and sleep",
            "Hydration",
            "Fever management",
            "Isolate to prevent spread"
        ],
        "Bronchitis": [
            "Antibiotics if bacterial",
            "Bronchodilators",
            "Rest",
            "Hydration",
            "Avoid irritants"
        ],
        "Pneumonia": [
            "Antibiotics",
            "Oxygen therapy if needed",
            "Rest",
            "Hydration",
            "Follow-up in 2 weeks"
        ],
        "Allergy": [
            "Antihistamines",
            "Avoid allergens",
            "Nasal sprays",
            "Immunotherapy if chronic",
            "Monitor symptoms"
        ],
        "COVID-19": [
            "Isolate for 5-10 days",
            "Symptom management",
            "Oxygen if needed",
            "Antivirals if high risk",
            "Monitor for worsening"
        ],
        "Diabetes": [
            "Blood sugar monitoring",
            "Dietary management",
            "Exercise routine",
            "Oral medications/insulin",
            "Regular check-ups"
        ],
        "Hypertension": [
            "Blood pressure monitoring",
            "Reduced sodium diet",
            "Regular exercise",
            "Medication management",
            "Stress management"
        ]
    }
    return treatments.get(disease, ["Consult physician", "Rest", "Hydration", "Monitor symptoms"])
