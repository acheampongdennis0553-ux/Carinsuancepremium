import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model and preprocessing objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title="Car Insurance Premium Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Insurance Premium Predictor")
st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Information")
    car_age = st.slider("Car Age (years)", min_value=0, max_value=14, value=5)
    car_value = st.number_input("Car Value (₹)", min_value=100000, max_value=3000000, value=1500000, step=50000)
    engine_cc = st.number_input("Engine CC", min_value=500, max_value=5000, value=1500, step=100)
    fuel_type = st.selectbox("Fuel Type", options=['Diesel', 'Electric', 'Hybrid', 'Petrol'])

with col2:
    st.subheader("Owner & Policy Information")
    transmission = st.selectbox("Transmission", options=['Automatic', 'Manual'])
    owner_age = st.slider("Owner Age (years)", min_value=18, max_value=80, value=40)
    ncb_percent = st.slider("No Claim Bonus (%)", min_value=0, max_value=50, value=25, step=5)
    accident_history = st.selectbox("Accident History", options=['No', 'Yes'])
    city_tier = st.selectbox("City Tier", options=['tier1', 'tier2', 'tier3'])

st.markdown("---")

# Prepare data for prediction
if st.button("🔮 Predict Premium", key="predict_btn"):
    try:
        # Create input dataframe with same structure as training data
        input_data = pd.DataFrame({
            'car_age_years': [car_age],
            'car_value': [car_value],
            'engine_cc': [engine_cc],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'owner_age': [owner_age],
            'ncb_percent': [ncb_percent],
            'accident_history': [accident_history],
            'city_tier': [city_tier]
        })
        
        # Encode categorical variables using the same label encoders
        categorical_columns = ['fuel_type', 'transmission', 'accident_history', 'city_tier']
        for col in categorical_columns:
            input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Estimated Annual Premium",
                value=f"₹{prediction:,.2f}",
                delta=None
            )
        
        with col2:
            # Calculate monthly equivalent
            monthly_premium = prediction / 12
            st.metric(
                label="Estimated Monthly Premium",
                value=f"₹{monthly_premium:,.2f}",
                delta=None
            )
        
        st.markdown("---")
        st.success("✅ Prediction completed successfully!")
        
        # Additional info
        st.info(
            f"**Input Summary:**\n\n"
            f"- Car Age: {car_age} years\n"
            f"- Car Value: ₹{car_value:,.0f}\n"
            f"- Engine: {engine_cc} CC\n"
            f"- Fuel Type: {fuel_type}\n"
            f"- Transmission: {transmission}\n"
            f"- Owner Age: {owner_age} years\n"
            f"- No Claim Bonus: {ncb_percent}%\n"
            f"- Accident History: {accident_history}\n"
            f"- City Tier: {city_tier}"
        )
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

st.markdown("---")
st.markdown(
    """
    ### 📈 About This Model
    
    This machine learning model predicts car insurance premiums based on:
    - Vehicle characteristics (age, value, engine specifications)
    - Owner information (age, claim history)
    - Policy factors (no claim bonus, city tier)
    
    **Model Performance:**
    - R² Score: 0.9192 (92% of variance explained)
    - RMSE: ₹5,797.50
    - MAE: ₹4,634.50
    
    **Key Feature Importance:**
    1. Car Value (90.5%)
    2. Car Age (4.3%)
    3. Accident History (1.6%)
    """
)
