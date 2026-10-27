import os
import unittest
import numpy as np
import pandas as pd
import joblib
from src.forecasting import EnergyForecaster

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

class TestEnergyForecastingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecaster = EnergyForecaster()
        cls.cache_path = os.path.join(OUTPUT_DIR, "dashboard_data.parquet")
        cls.assertTrue(os.path.exists(cls.cache_path), "Dashboard data cache missing")
        cls.df = pd.read_parquet(cls.cache_path)

    def test_model_loaded(self):
        self.assertIsNotNone(self.forecaster.model, "Champion model failed to load")
        self.assertIsNotNone(self.forecaster.meta, "Model metadata failed to load")
        self.assertGreater(len(self.forecaster.feature_cols), 10, "Feature list too short")

    def test_forecast_horizon_24(self):
        res = self.forecaster.forecast_horizon(self.df, meter_id=4, start_timestamp='2022-03-01 00:00:00', horizon_hours=24)
        self.assertEqual(len(res), 24, "Forecast horizon did not return exactly 24 hours")
        self.assertTrue((res['forecast_kwh'] >= 0).all(), "Negative consumption forecasted")
        self.assertTrue((res['upper_bound_kwh'] >= res['lower_bound_kwh']).all(), "Confidence bounds inverted")

    def test_forecast_horizon_48(self):
        res = self.forecaster.forecast_horizon(self.df, meter_id=25, start_timestamp='2022-02-15 00:00:00', horizon_hours=48)
        self.assertEqual(len(res), 48, "Forecast horizon did not return 48 hours")

    def test_invalid_meter_error_handling(self):
        with self.assertRaises(ValueError):
            self.forecaster.forecast_horizon(self.df, meter_id=99999, start_timestamp='2022-03-01', horizon_hours=24)

    def test_single_instance_prediction(self):
        sample_dict = {col: 1.0 for col in self.forecaster.feature_cols}
        sample_dict['air_temperature'] = 28.5
        sample_dict['lag_24'] = 65.0
        sample_dict['rolling_mean_24h_lag24'] = 58.0
        out = self.forecaster.predict_instance(sample_dict)
        self.assertIn('predicted_consumption_kwh', out)
        self.assertGreaterEqual(out['predicted_consumption_kwh'], 0.0)

    def test_shap_artifacts_exist(self):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, "shap_explainer.joblib")), "SHAP explainer missing")
        self.assertTrue(os.path.exists(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")), "SHAP importance missing")
        self.assertTrue(os.path.exists(os.path.join(OUTPUT_DIR, "shap_beeswarm_summary.png")), "SHAP beeswarm missing")

if __name__ == "__main__":
    unittest.main()
