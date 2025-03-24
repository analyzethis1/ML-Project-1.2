# Copyright (c) 2025 Chris Karim
# All rights reserved.
#
# This source code is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# See LICENSE file or visit https://creativecommons.org/licenses/by-nc-nd/4.0/

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import datetime
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load model and scaler
model = joblib.load("best_lightgbm_model.pkl")
scaler = joblib.load("scaler_lgb.pkl")

# --- STREAMLIT APP ---
st.title("🏭 HVAC Repair Prediction: Energy Forecast Dashboard")

st.markdown("Upload your batch CSV with timestamp, building_id, site_id, and sensor data:")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Batch file uploaded. Beginning processing...")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['temp_diff'] = df['air_temperature'] - df['dew_temperature']
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['temp_diff_x_square_feet'] = df['temp_diff'] * df['square_feet']
    df['building_temp_diff'] = df['air_temperature'] * df['square_feet']
    df['site_hour'] = df['site_id'].astype(int) * df['hour']

    df['meter_reading_lag_1h'] = df.groupby('building_id')['meter_reading'].shift(1).fillna(0)
    df['meter_reading_lag_24h'] = df.groupby('building_id')['meter_reading'].shift(24).fillna(0)
    df['meter_reading_lag_7d'] = df.groupby('building_id')['meter_reading'].shift(24*7).fillna(0)
    df['hourly_meter_change'] = df['meter_reading'].diff().fillna(0)

    # Log-transform target (for inverse later)
    df['meter_reading'] = df['meter_reading'].clip(lower=1e-5)
    y_true = df['meter_reading'].copy()
    df['meter_reading'] = np.log1p(df['meter_reading'])

    df['building_id'] = df['building_id'].astype('category')
    df['site_id'] = df['site_id'].astype('category')

    feature_cols = [
        'air_temperature', 'dew_temperature', 'square_feet', 'floor_count',
        'sin_hour', 'cos_hour', 'month', 'week_of_year',
        'meter_reading_lag_1h', 'meter_reading_lag_24h', 'meter_reading_lag_7d',
        'temp_diff', 'hourly_meter_change', 'temp_diff_x_square_feet',
        'building_temp_diff', 'site_hour', 'building_id', 'site_id'
    ]

    numeric_cols = [col for col in feature_cols if df[col].dtype != 'category']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    preds = model.predict(df[feature_cols])
    preds = np.expm1(preds)
    y_true = y_true.reset_index(drop=True)

    df['predicted_meter_reading'] = preds
    df['residual'] = y_true - preds

    # Binary repair signal based on high residuals (thresholded)
    residual_threshold = df['residual'].std() * 2
    df['repair_flag'] = (np.abs(df['residual']) > residual_threshold).astype(int)

    st.markdown("### Prediction Scatter Plot (Actual vs Predicted)")
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true[:5000], y=preds[:5000], hue=df['site_id'][:5000], palette='viridis', alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='gray')
    plt.xlabel("Actual Meter Reading")
    plt.ylabel("Predicted Meter Reading")
    plt.title("Actual vs Predicted | Color by Site ID")
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown("### Residual Distribution")
    plt.figure(figsize=(10, 4))
    sns.histplot(df['residual'], bins=50, kde=True, color="coral")
    plt.axvline(residual_threshold, color='red', linestyle='--', label='+2σ')
    plt.axvline(-residual_threshold, color='red', linestyle='--')
    plt.title("Residuals (Actual - Predicted) with Repair Threshold")
    plt.xlabel("Residual")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown("### Repair Signal Preview")
    st.dataframe(df[['timestamp', 'building_id', 'site_id', 'predicted_meter_reading', 'residual', 'repair_flag']].head(10))

    st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
