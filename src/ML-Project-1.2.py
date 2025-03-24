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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import optuna

print("Starting script...")
dataset_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Datasets", "ashrae-energy-prediction")
metadata = pd.read_csv(os.path.join(dataset_folder, "building_metadata.csv"))
train = pd.read_csv(os.path.join(dataset_folder, "train.csv"))
weather_train = pd.read_csv(os.path.join(dataset_folder, "weather_train.csv"))

print("Loaded CSVs. Full dataset in use")

# Merge & Feature Engineering
df = pd.merge(train, metadata, on='building_id', how='left')
df = pd.merge(df, weather_train, on=['site_id', 'timestamp'], how='left')
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
df['meter_reading'] = df['meter_reading'].clip(lower=1e-5)
df['meter_reading_lag_1h'] = df.groupby('building_id')['meter_reading'].shift(1).fillna(0)
df['meter_reading_lag_24h'] = df.groupby('building_id')['meter_reading'].shift(24).fillna(0)
df['meter_reading_lag_7d'] = df.groupby('building_id')['meter_reading'].shift(24*7).fillna(0)
df['hourly_meter_change'] = df['meter_reading'].diff().fillna(0)

# Log-transform target
df['meter_reading'] = np.log1p(df['meter_reading'])

# Fill missing numeric values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Add categorical features
df['building_id'] = df['building_id'].astype('category')
df['site_id'] = df['site_id'].astype('category')

# Feature list
feature_cols = [
    'air_temperature', 'dew_temperature', 'square_feet', 'floor_count',
    'sin_hour', 'cos_hour', 'month', 'week_of_year',
    'meter_reading_lag_1h', 'meter_reading_lag_24h', 'meter_reading_lag_7d',
    'temp_diff', 'hourly_meter_change', 'temp_diff_x_square_feet',
    'building_temp_diff', 'site_hour', 'building_id', 'site_id'
]

# Normalize features (except categorical)
numeric_features = [col for col in feature_cols if df[col].dtype != 'category']
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
joblib.dump(scaler, 'scaler_lgb.pkl')

X = df[feature_cols]
y = df['meter_reading']

print("Running Optuna LightGBM tuning manually...")

def objective(trial):
    param_grid = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

    cv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['building_id', 'site_id'])
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=['building_id', 'site_id'])
        model = lgb.train(param_grid, train_data, valid_sets=[val_data], num_boost_round=1000, callbacks=[lgb.early_stopping(50)])
        preds = model.predict(X_val)
        score = mean_squared_error(np.expm1(y_val), np.expm1(preds), squared=False)
        scores.append(score)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)

print("Best params:", study.best_params)

joblib.dump(study, 'optuna_study.pkl')

best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt'
})

# Evaluate top 3 trials
for i, trial in enumerate(study.best_trials[:3]):
    print(f"\nEvaluating Trial #{trial.number} with value {trial.value:.2f}")
    params = trial.params.copy()
    params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    })
    model = lgb.train(
        params,
        lgb.Dataset(X, label=y, categorical_feature=['building_id', 'site_id']),
        num_boost_round=1000
    )
    y_pred = model.predict(X)
    y_exp = np.expm1(y)
    y_pred_exp = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_exp, y_pred_exp))
    mae = mean_absolute_error(y_exp, y_pred_exp)
    r2 = r2_score(y_exp, y_pred_exp)

    print(f'Trial {trial.number} | RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}')

    if i == 0:
        joblib.dump(model, 'best_lightgbm_model.pkl')
        plt.figure(figsize=(6, 6))
        plt.scatter(y_exp[:5000], y_pred_exp[:5000], alpha=0.4)
        plt.xlabel('Actual Meter Reading')
        plt.ylabel('Predicted Meter Reading')
        plt.title(f'Trial {trial.number} | Actual vs Predicted')
        plt.grid(True)
        plt.plot([y_exp.min(), y_exp.max()], [y_exp.min(), y_exp.max()], '--', color='gray')
        plt.tight_layout()
        plt.savefig('pred_vs_actual_trial_{}.png'.format(trial.number))
        plt.show()

    if i == 0:
        joblib.dump(model, 'best_lightgbm_model.pkl')

        high_usage_mask = y_exp > 1_000_000
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=y_exp[high_usage_mask],
            y=y_pred_exp[high_usage_mask],
            hue=df.loc[high_usage_mask, 'site_id'],
            palette='viridis', alpha=0.6, edgecolor=None
        )
        sns.regplot(
            x=y_exp[high_usage_mask],
            y=y_pred_exp[high_usage_mask],
            scatter=False, ci=None, color='black', line_kws={'linestyle': '--'}
        )
        plt.plot([y_exp.min(), y_exp.max()], [y_exp.min(), y_exp.max()], '--', color='gray')
        plt.xlabel('Actual Meter Reading')
        plt.ylabel('Predicted Meter Reading')
        plt.title(f'Trial {trial.number} | High Usage >1M: Actual vs Predicted by Site')
        plt.legend(title='Site ID')
        plt.tight_layout()
        plt.savefig(f'high_usage_actual_vs_pred_trial_{trial.number}.png')
        plt.show()

        # Residual Plot
        residuals = y_exp[high_usage_mask] - y_pred_exp[high_usage_mask]
        plt.figure(figsize=(8, 4))
        sns.histplot(residuals, bins=50, kde=True, color='coral')
        plt.title('Residuals for High Usage Buildings (Actual - Predicted)')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'high_usage_residuals_trial_{trial.number}.png')
        plt.show()
