import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

class EnergyForecaster:
    def __init__(self, model_name='lightgbm_forecaster.joblib'):
        self.model_path = os.path.join(MODELS_DIR, model_name)
        self.meta_path = os.path.join(MODELS_DIR, "feature_metadata.json")
        self.model = None
        self.meta = None
        self.rmse = 12.71  # Default test RMSE for confidence interval calculation
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta['feature_cols']
        self.rmse = self.meta.get('champion_test_rmse', 12.71)

    def forecast_horizon(self, df_full, meter_id, start_timestamp, horizon_hours=24):
        """
        Generates day-ahead or multi-horizon forecast for a specific building meter.
        """
        # Filter for the meter
        b_df = df_full[df_full['meter_id'] == meter_id].sort_values('timestamp').reset_index(drop=True)
        if len(b_df) == 0:
            raise ValueError(f"No records found for meter_id={meter_id}")

        # Find starting index
        start_ts = pd.to_datetime(start_timestamp)
        sub = b_df[b_df['timestamp'] >= start_ts].head(horizon_hours)

        if len(sub) == 0:
            # Fall back to latest available
            sub = b_df.tail(horizon_hours)

        X = sub[self.feature_cols]
        preds = self.model.predict(X)
        preds = np.maximum(0, preds)

        # 95% prediction interval (1.96 * RMSE)
        lower_bound = np.maximum(0, preds - 1.96 * self.rmse)
        upper_bound = preds + 1.96 * self.rmse

        result_df = pd.DataFrame({
            'timestamp': sub['timestamp'].values,
            'meter_id': meter_id,
            'forecast_kwh': np.round(preds, 2),
            'lower_bound_kwh': np.round(lower_bound, 2),
            'upper_bound_kwh': np.round(upper_bound, 2),
            'actual_kwh': np.round(sub['consumption_kwh'].values, 2) if 'consumption_kwh' in sub.columns else None,
            'air_temperature': sub['air_temperature'].values if 'air_temperature' in sub.columns else None,
            'category': sub['category'].values[0] if 'category' in sub.columns else 'Unknown'
        })
        return result_df

    def predict_instance(self, feature_dict):
        """
        Predicts consumption for a single feature vector dictionary.
        """
        row = pd.DataFrame([feature_dict])
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0.0
        X = row[self.feature_cols]
        pred = float(self.model.predict(X)[0])
        pred = max(0.0, pred)
        return {
            'predicted_consumption_kwh': round(pred, 2),
            'lower_bound_kwh': round(max(0.0, pred - 1.96 * self.rmse), 2),
            'upper_bound_kwh': round(pred + 1.96 * self.rmse, 2)
        }
