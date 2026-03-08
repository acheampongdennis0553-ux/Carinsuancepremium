# Car Insurance Premium Prediction Model

A machine learning application that predicts car insurance premiums based on car and owner details.

## Features

- **Machine Learning Model**: Random Forest Regressor trained on car insurance dataset
- **Web Interface**: Interactive Streamlit application for easy predictions
- **Input Parameters**: 
  - Car Age, Car Value, Engine CC
  - Fuel Type, Transmission, Owner Age
  - NCB Bonus %, Accident History, City Tier

## Installation

```bash
pip install -r requirements.txt
```

## Training the Model

```bash
python train_model.py
```

This will create:
- `insurance_model.pkl`: Trained Random Forest model
- `label_encoders.pkl`: Categorical variable encoders

## Running the App Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment on Streamlit Cloud

### Steps:

1. Ensure the code is pushed to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository: `acheampongdennis0553-ux/Carinsuancepremium`
5. Select branch: `main`
6. Select file: `app.py`
7. Click "Deploy"

## Files Structure

- `app.py` - Main Streamlit application
- `train_model.py` - Model training script
- `requirements.txt` - Python dependencies
- `insurance_model.pkl` - Trained model (binary)
- `label_encoders.pkl` - Category encoders (binary)
- `car_insurance_premium_regression_dataset (1) (1).csv` - Training dataset

## Model Performance

- Algorithm: Random Forest Regressor
- Training samples: 302
- Features: 9

## Author

acheampongdennis0553-ux
