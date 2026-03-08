import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('car_insurance_premium_regression_dataset (1) (1).csv')

# Handle missing values
df = df.dropna(subset=['annual_car_premium'])  # Drop if target is missing
df['car_value'].fillna(df['car_value'].median(), inplace=True)
df['engine_cc'].fillna(df['engine_cc'].median(), inplace=True)
df['owner_age'].fillna(df['owner_age'].median(), inplace=True)
df['ncb_percent'].fillna(df['ncb_percent'].median(), inplace=True)
df['fuel_type'].fillna(df['fuel_type'].mode()[0], inplace=True)
df['transmission'].fillna(df['transmission'].mode()[0], inplace=True)
df['accident_history'].fillna(df['accident_history'].mode()[0], inplace=True)
df['city_tier'].fillna(df['city_tier'].mode()[0], inplace=True)

# Separate features and target
X = df.drop('annual_car_premium', axis=1)
y = df['annual_car_premium']

# Encode categorical variables
label_encoders = {}
categorical_columns = ['fuel_type', 'transmission', 'accident_history', 'city_tier']

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Save model and preprocessing objects
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("\nModel and preprocessing objects saved!")
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)
