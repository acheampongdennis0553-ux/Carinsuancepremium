import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="Car Insurance Premium Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Car Insurance Premium Predictor")
st.markdown("---")

# Cache the model training
@st.cache_resource
def train_model():
    """Train and return the model and encoders"""
    try:
        # Load the CSV file
        csv_file = 'car_insurance_premium_regression_dataset (1) (1).csv'
        
        if not os.path.exists(csv_file):
            st.error(f"Data file not found: {csv_file}")
            st.stop()
        
        df = pd.read_csv(csv_file)
        
        # Data preprocessing
        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('unknown', inplace=True)
        
        # Separate features and target
        if 'annual_car_premium' not in df.columns:
            st.error("Target column 'annual_car_premium' not found in dataset")
            st.stop()
        
        X = df.drop('annual_car_premium', axis=1)
        y = df['annual_car_premium']
        
        # Encode categorical variables
        encoders = {}
        categorical_features = ['fuel_type', 'transmission', 'accident_history', 'city_tier']
        
        for col in categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le
        
        # Train the model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
        model.fit(X, y)
        
        return model, encoders, X.columns.tolist()
    
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        st.stop()

# Load model
try:
    model, encoders, feature_names = train_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

st.markdown("---")

# Create input form
st.subheader("Enter Car and Owner Details")

col1, col2, col3 = st.columns(3)

with col1:
    car_age = st.number_input(
        "Car Age (years)",
        min_value=0,
        max_value=30,
        value=5,
        step=1
    )
    
    car_value = st.number_input(
        "Car Value (Rupees)",
        min_value=100000,
        max_value=5000000,
        value=1000000,
        step=50000
    )
    
    engine_cc = st.number_input(
        "Engine CC",
        min_value=600,
        max_value=2500,
        value=1500,
        step=100
    )

with col2:
    fuel_type = st.selectbox(
        "Fuel Type",
        ["petrol", "diesel", "hybrid", "electric"]
    )
    
    transmission = st.selectbox(
        "Transmission",
        ["manual", "automatic"]
    )
    
    owner_age = st.number_input(
        "Owner Age (years)",
        min_value=18,
        max_value=80,
        value=35,
        step=1
    )

with col3:
    ncb_percent = st.number_input(
        "NCB (No Claim Bonus) %",
        min_value=0,
        max_value=100,
        value=20,
        step=5
    )
    
    accident_history = st.selectbox(
        "Accident History",
        ["no", "yes"]
    )
    
    city_tier = st.selectbox(
        "City Tier",
        ["tier1", "tier2", "tier3"]
    )

st.markdown("---")

# Make prediction
if st.button("Calculate Premium", type="primary"):
    try:
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
            if col in encoders:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except Exception as e:
                    st.error(f"Error encoding {col}: {str(e)}")
                    st.stop()
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        
        # Display result
        col_result1, col_result2 = st.columns([1, 1])
        
        with col_result1:
            st.metric(
                "Estimated Annual Premium",
                f"Rs {prediction:,.2f}"
            )
        
        # Display input summary
        st.markdown("---")
        st.subheader("Summary of Inputs")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Car Age", f"{car_age} years")
            st.metric("Car Value", f"Rs {car_value:,}")
            st.metric("Engine CC", f"{engine_cc}")
        
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
        import traceback
        st.write(traceback.format_exc())

st.markdown("---")
st.info("This model predicts your car insurance premium based on your vehicle and personal details.")

