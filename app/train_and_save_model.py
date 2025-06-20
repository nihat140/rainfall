import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load and clean data (from your notebook)
data = pd.read_csv("ken-rainfall-adm2-full.csv")
data_cleaned = data.iloc[1:].copy()
numeric_columns = ['n_pixels', 'rfh', 'rfh_avg', 'r1h', 'r1h_avg', 'r3h', 'r3h_avg', 'rfq', 'r1q', 'r3q']
for col in numeric_columns:
    data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
data_cleaned['date'] = pd.to_datetime(data_cleaned['date'], errors='coerce')
columns_to_fill = ['r1h', 'r1h_avg', 'r3h', 'r3h_avg', 'r1q', 'r3q']
for col in columns_to_fill:
    data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].mean())

# Feature engineering
rf_data = data_cleaned[['date', 'rfh']].copy()
rf_data['year'] = rf_data['date'].dt.year
rf_data['month'] = rf_data['date'].dt.month
rf_data['rfh_lag1'] = rf_data['rfh'].shift(1)
rf_data = rf_data.dropna()

# Split data
X = rf_data[['year', 'month', 'rfh_lag1']]
y = rf_data['rfh']
X_train, X_test = X[:-12], X[-12:]
y_train, y_test = y[:-12], y[-12:]

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
joblib.dump(rf_model, 'rf_model.pkl')

# Evaluate and print MAE
predictions = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model trained and saved. MAE: {mae:.2f}")