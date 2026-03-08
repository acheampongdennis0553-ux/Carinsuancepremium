import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('car_insurance_premium_regression_dataset (1) (1).csv')

print('Shape:', df.shape)
print('\nColumn names:')
print(df.columns.tolist())
print('\nFirst 5 rows:')
print(df.head())
print('\nData types:')
print(df.dtypes)
print('\nMissing values:')
print(df.isnull().sum())
print('\nBasic statistics:')
print(df.describe())
