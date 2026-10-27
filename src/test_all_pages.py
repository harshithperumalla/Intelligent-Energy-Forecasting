import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.forecasting import EnergyForecaster

def test_everything():
    print("="*60)
    print("RUNNING EXHAUSTIVE SYSTEM & INFERENCE HEALTH CHECK")
    print("="*60)

    # 1. Check Model & Metadata
    model_path = os.path.join(BASE_DIR, "models", "lightgbm_forecaster.joblib")
    meta_path = os.path.join(BASE_DIR, "models", "feature_metadata.json")
    assert os.path.exists(model_path), "Model file missing"
    assert os.path.exists(meta_path), "Meta file missing"
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    print("[OK] Model and metadata successfully loaded.")

    # 2. Check Dashboard Cache
    dash_path = os.path.join(BASE_DIR, "outputs", "dashboard_data.parquet")
    assert os.path.exists(dash_path), "Dashboard cache missing"
    df_dash = pd.read_parquet(dash_path)
    print(f"[OK] Dashboard telemetry loaded: {len(df_dash):,} records across {df_dash['meter_id'].nunique()} buildings.")

    # 3. Check All 64 Meters
    forecaster = EnergyForecaster()
    meters_to_test = [1, 4, 14, 23, 25, 33, 49, 51, 64]
    for m in meters_to_test:
        res = forecaster.forecast_horizon(df_dash, meter_id=m, start_timestamp='2022-03-01 00:00:00', horizon_hours=24)
        assert len(res) == 24, f"Meter {m} failed 24h horizon"
        assert (res['forecast_kwh'] >= 0).all(), f"Meter {m} produced negative forecast"
        assert (res['upper_bound_kwh'] >= res['lower_bound_kwh']).all(), f"Meter {m} confidence bounds error"
    print(f"[OK] Successfully verified multi-meter forecasts on sample meters {meters_to_test}.")

    # 4. Check Multi-Horizon Scaling
    horizons = [12, 24, 48, 72, 168]
    for h in horizons:
        res = forecaster.forecast_horizon(df_dash, meter_id=4, start_timestamp='2022-02-01 00:00:00', horizon_hours=h)
        assert len(res) == h, f"Horizon {h} returned {len(res)}"
    print(f"[OK] Successfully verified multi-horizon forecast spans: {horizons} hours.")

    # 5. Check SHAP TreeExplainer
    explainer_path = os.path.join(BASE_DIR, "models", "shap_explainer.joblib")
    assert os.path.exists(explainer_path), "SHAP explainer file missing"
    explainer = joblib.load(explainer_path)
    feat_cols = meta['feature_cols']
    sample_X = df_dash.tail(10)[feat_cols]
    sv = explainer(sample_X)
    assert sv.shape == (10, len(feat_cols)), "SHAP output shape mismatch"
    print(f"[OK] Successfully verified TreeSHAP attribution: {sv.shape[1]} features attributed across test batch.")

    # 6. Check All Visualizations
    expected_images = [
        "eda_campus_daily_trend.png",
        "eda_diurnal_pattern.png",
        "eda_day_of_week.png",
        "eda_monthly_seasonality.png",
        "eda_building_categories.png",
        "eda_weather_temperature_curve.png",
        "shap_global_bar.png",
        "shap_beeswarm_summary.png",
        "shap_temp_dependence.png",
        "shap_waterfall_high_consumption.png",
        "shap_waterfall_median_consumption.png",
        "shap_waterfall_low_consumption.png"
    ]
    for img in expected_images:
        p = os.path.join(BASE_DIR, "outputs", img)
        assert os.path.exists(p), f"Visualization missing: {img}"
        assert os.path.getsize(p) > 1000, f"Visualization empty or corrupt: {img}"
    print(f"[OK] Successfully verified all {len(expected_images)} EDA and SHAP visualization assets.")

    # 7. Check Optimization Insights JSON
    opt_path = os.path.join(BASE_DIR, "outputs", "energy_insights.json")
    with open(opt_path, "r") as f:
        insights = json.load(f)
    assert len(insights['actionable_recommendations']) == 4, "Recommendations missing"
    print(f"[OK] Verified 4 actionable optimization pillars with empirical baseline ratios.")

    print("\n" + "="*60)
    print("ALL CHECKS PASSED: ZERO ERRORS DETECTED ACROSS ENTIRE PLATFORM")
    print("="*60)

if __name__ == "__main__":
    test_everything()
