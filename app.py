import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Car Insurance Premium Predictor", layout="wide")

st.title("🚗 Car Insurance Premium Predictor")
st.markdown("---")

# Load model and encoders
try:
    with open('insurance_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# Create input columns
col1, col2, col3 = st.columns(3)

with col1:
    car_age = st.number_input("Car Age (years)", min_value=0, max_value=30, value=5)
    car_value = st.number_input("Car Value (₹)", min_value=100000, max_value=5000000, value=1000000, step=50000)
    engine_cc = st.number_input("Engine CC", min_value=600, max_value=2500, value=1500, step=100)

with col2:
    fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel", "hybrid", "electric"])
    transmission = st.selectbox("Transmission", ["manual", "automatic"])
    owner_age = st.number_input("Owner Age (years)", min_value=18, max_value=80, value=35)

with col3:
    ncb_percent = st.number_input("NCB % (No Claim Bonus)", min_value=0, max_value=100, value=20, step=5)
    accident_history = st.selectbox("Accident History", ["no", "yes"])
    city_tier = st.selectbox("City Tier", ["tier1", "tier2", "tier3"])

# Prepare input data
input_data = {
    'car_age_years': [car_age],
    'car_value': [car_value],
    'engine_cc': [engine_cc],
    'fuel_type': [fuel_type],
    'transmission': [transmission],
    'owner_age': [owner_age],
    'ncb_percent': [ncb_percent],
    'accident_history': [accident_history],
    'city_tier': [city_tier]
}

input_df = pd.DataFrame(input_data)

# Encode categorical variables
for col in ['fuel_type', 'transmission', 'accident_history', 'city_tier']:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Make prediction
if st.button("🔮 Predict Premium", key="predict"):
    try:
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        st.success(f"### Estimated Annual Premium: ₹{prediction:,.2f}")
        
        # Display input summary
        st.subheader("Input Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Car Age", f"{car_age} years")
            st.metric("Car Value", f"₹{car_value:,}")
            st.metric("Engine CC", f"{engine_cc} cc")
        
        with summary_col2:
            st.metric("Fuel Type", fuel_type.upper())
            st.metric("Transmission", transmission.upper())
            st.metric("Owner Age", f"{owner_age} years")
        
        with summary_col3:
            st.metric("NCB Bonus", f"{ncb_percent}%")
            st.metric("Accident History", accident_history.upper())
            st.metric("City Tier", city_tier.upper())
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.markdown("---")
st.info("💡 This model predicts your car insurance premium based on your car and personal details.")
