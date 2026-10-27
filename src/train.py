import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    # Temporal & Cyclical
    'hour', 'dayofweek', 'month', 'day', 'dayofyear', 'is_weekend', 'is_business_hours',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    # Calendar
    'is_holiday', 'is_semester', 'is_exam', 'event_count',
    # Weather & Thermal
    'air_temperature', 'apparent_temperature', 'relative_humidity', 'wind_speed',
    'cooling_degree_hours', 'heating_degree_hours', 'temp_humidity_index',
    # Building Metadata
    'campus_id', 'meter_id', 'category_code', 'gross_floor_area', 'built_year', 'capacity',
    # Leakage-Free Autoregressive Lags & Rolling (>= 24h)
    'lag_24', 'lag_48', 'lag_168',
    'rolling_mean_24h_lag24', 'rolling_std_24h_lag24', 'rolling_max_24h_lag24', 'rolling_min_24h_lag24'
]

TARGET_COL = 'consumption_kwh'

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Safe MAPE on non-zero consumption
    non_zero_mask = y_true > 1.0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100 if non_zero_mask.sum() > 0 else 0.0
    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'R2': round(float(r2), 4),
        'MAPE_pct': round(float(mape), 2)
    }

def train_and_evaluate_models():
    print("="*60)
    print("STARTING MODEL TRAINING & EVALUATION PIPELINE")
    print("="*60)
    t_start = time.time()

    data_path = os.path.join(DATA_DIR, "featured_hourly_consumption.parquet")
    print(f"Loading feature-engineered dataset: {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records.")

    # Chronological Split
    # Train: < 2021-07-01
    # Val: 2021-07-01 to 2021-12-31
    # Test: >= 2022-01-01
    print("\nSplitting chronologically into Train, Validation, and Test sets...")
    train_mask = df['timestamp'] < '2021-07-01'
    val_mask = (df['timestamp'] >= '2021-07-01') & (df['timestamp'] < '2022-01-01')
    test_mask = df['timestamp'] >= '2022-01-01'

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    print(f"Train set: {len(train_df):,} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Val set:   {len(val_df):,} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"Test set:  {len(test_df):,} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL].values
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL].values
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL].values

    # Fit Scaler for linear baseline
    print("\nFitting StandardScaler for linear baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    results = {}
    test_predictions = {
        'timestamp': test_df['timestamp'].values,
        'meter_id': test_df['meter_id'].values,
        'campus_id': test_df['campus_id'].values,
        'actual': y_test
    }

    # 1. Baseline: Persistent Seasonal Naive (Lag 168 - Same hour last week)
    print("\nEvaluating Baseline 1: Persistent Seasonal Naive (Lag 168)...")
    y_pred_naive_val = val_df['lag_168'].values
    y_pred_naive_test = test_df['lag_168'].values
    results['Baseline_Seasonal_Naive'] = {
        'Validation': evaluate_metrics(y_val, y_pred_naive_val),
        'Test': evaluate_metrics(y_test, y_pred_naive_test)
    }
    test_predictions['pred_naive'] = y_pred_naive_test
    print(f"Naive Test Metrics: {results['Baseline_Seasonal_Naive']['Test']}")

    # 2. Linear Baseline: Ridge Regression
    print("\nTraining Model 1: Ridge Regression...")
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge_val = np.maximum(0, ridge.predict(X_val_scaled))
    y_pred_ridge_test = np.maximum(0, ridge.predict(X_test_scaled))
    results['Ridge_Regression'] = {
        'Validation': evaluate_metrics(y_val, y_pred_ridge_val),
        'Test': evaluate_metrics(y_test, y_pred_ridge_test)
    }
    test_predictions['pred_ridge'] = y_pred_ridge_test
    joblib.dump(ridge, os.path.join(MODELS_DIR, "ridge_forecaster.joblib"))
    print(f"Ridge Test Metrics: {results['Ridge_Regression']['Test']}")

    # 3. Random Forest Regressor
    print("\nTraining Model 2: Random Forest Regressor...")
    # Sample 200,000 training points for RF computational efficiency & fast convergence
    rf_sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(200000, len(X_train)), replace=False)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.8,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train.iloc[rf_sample_idx], y_train[rf_sample_idx])
    y_pred_rf_val = np.maximum(0, rf.predict(X_val))
    y_pred_rf_test = np.maximum(0, rf.predict(X_test))
    results['Random_Forest'] = {
        'Validation': evaluate_metrics(y_val, y_pred_rf_val),
        'Test': evaluate_metrics(y_test, y_pred_rf_test)
    }
    test_predictions['pred_rf'] = y_pred_rf_test
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_forecaster.joblib"))
    print(f"Random Forest Test Metrics: {results['Random_Forest']['Test']}")

    # 4. Gradient Boosting: LightGBM Regressor (Primary Champion Model)
    print("\nTraining Model 3: LightGBM Regressor (Champion Model)...")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=600,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )

    y_pred_lgb_val = np.maximum(0, lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration))
    y_pred_lgb_test = np.maximum(0, lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration))
    results['LightGBM_Champion'] = {
        'Validation': evaluate_metrics(y_val, y_pred_lgb_val),
        'Test': evaluate_metrics(y_test, y_pred_lgb_test)
    }
    test_predictions['pred_lgbm'] = y_pred_lgb_test
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, "lightgbm_forecaster.joblib"))
    print(f"LightGBM Test Metrics: {results['LightGBM_Champion']['Test']}")

    # 5. XGBoost Regressor Comparison
    print("\nTraining Model 4: XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    # Train on representative sample of 400,000 for rapid execution
    xgb_sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(400000, len(X_train)), replace=False)
    xgb_reg.fit(X_train.iloc[xgb_sample_idx], y_train[xgb_sample_idx])
    y_pred_xgb_val = np.maximum(0, xgb_reg.predict(X_val))
    y_pred_xgb_test = np.maximum(0, xgb_reg.predict(X_test))
    results['XGBoost'] = {
        'Validation': evaluate_metrics(y_val, y_pred_xgb_val),
        'Test': evaluate_metrics(y_test, y_pred_xgb_test)
    }
    test_predictions['pred_xgboost'] = y_pred_xgb_test
    joblib.dump(xgb_reg, os.path.join(MODELS_DIR, "xgboost_forecaster.joblib"))
    print(f"XGBoost Test Metrics: {results['XGBoost']['Test']}")

    # 6. Time-Series Cross Validation Stability (Step 9)
    print("\nValidating Forecast Stability with Time-Series Cross-Validation (3 folds)...")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_records = []
    # Use LightGBM on a downsampled time series for fast CV verification
    cv_sample_df = df.sample(frac=0.15, random_state=42).sort_values('timestamp')
    X_cv, y_cv = cv_sample_df[FEATURE_COLS], cv_sample_df[TARGET_COL].values

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        fold_X_tr, fold_y_tr = X_cv.iloc[train_idx], y_cv[train_idx]
        fold_X_va, fold_y_va = X_cv.iloc[val_idx], y_cv[val_idx]
        
        cv_lgb = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        cv_lgb.fit(fold_X_tr, fold_y_tr)
        preds = np.maximum(0, cv_lgb.predict(fold_X_va))
        fold_metrics = evaluate_metrics(fold_y_va, preds)
        fold_metrics['fold'] = fold + 1
        cv_records.append(fold_metrics)
        print(f"  Fold {fold+1}: MAE={fold_metrics['MAE']}, RMSE={fold_metrics['RMSE']}, R2={fold_metrics['R2']}")

    cv_df = pd.DataFrame(cv_records)
    cv_summary = {
        'folds': cv_records,
        'mean_MAE': round(float(cv_df['MAE'].mean()), 4),
        'std_MAE': round(float(cv_df['MAE'].std()), 4),
        'mean_RMSE': round(float(cv_df['RMSE'].mean()), 4),
        'std_RMSE': round(float(cv_df['RMSE'].std()), 4),
        'mean_R2': round(float(cv_df['R2'].mean()), 4),
        'std_R2': round(float(cv_df['R2'].std()), 4)
    }

    # Save Metadata and Results
    print("\nSaving test predictions and evaluation comparison tables...")
    test_pred_df = pd.DataFrame(test_predictions)
    test_pred_df.to_parquet(os.path.join(OUTPUT_DIR, "test_predictions.parquet"), index=False)

    comparison_rows = []
    for model_name, m_dict in results.items():
        comparison_rows.append({
            'Model': model_name,
            'Val_MAE': m_dict['Validation']['MAE'],
            'Val_RMSE': m_dict['Validation']['RMSE'],
            'Val_R2': m_dict['Validation']['R2'],
            'Val_MAPE_%': m_dict['Validation']['MAPE_pct'],
            'Test_MAE': m_dict['Test']['MAE'],
            'Test_RMSE': m_dict['Test']['RMSE'],
            'Test_R2': m_dict['Test']['R2'],
            'Test_MAPE_%': m_dict['Test']['MAPE_pct']
        })
    comp_df = pd.DataFrame(comparison_rows).sort_values('Test_RMSE')
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_table.csv"), index=False)
    print("\nModel Comparison Table:")
    print(comp_df.to_string(index=False))

    with open(os.path.join(OUTPUT_DIR, "model_comparison_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "cv_stability_results.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    # Save feature metadata for inference and SHAP explainability
    feature_meta = {
        'feature_cols': FEATURE_COLS,
        'target_col': TARGET_COL,
        'num_features': len(FEATURE_COLS),
        'best_iteration': int(lgb_model.best_iteration),
        'trained_timestamp': pd.Timestamp.now().isoformat(),
        'champion_model': 'LightGBM_Champion',
        'champion_test_r2': results['LightGBM_Champion']['Test']['R2'],
        'champion_test_mae': results['LightGBM_Champion']['Test']['MAE'],
        'champion_test_rmse': results['LightGBM_Champion']['Test']['RMSE']
    }
    with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    print(f"\nAll models trained and saved! Entire pipeline took {time.time() - t_start:.2f}s.")
    return results

if __name__ == "__main__":
    train_and_evaluate_models()
