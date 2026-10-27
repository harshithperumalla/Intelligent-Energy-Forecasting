# Intelligent Energy Consumption Forecasting for Smart Buildings (EnergiSense AI)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Champion_Model-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainable_AI-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit_1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, end-to-end Machine Learning and Explainable AI system for predictive electricity consumption forecasting across multi-facility university campuses and commercial buildings. Built using the UNICON Smart Building Energy & Utilities dataset.

---

## 📌 Executive Summary & Problem Formulation

- **Prediction Target**: Hourly Electricity Consumption ($kWh/h$).
- **Prediction Interval**: 1 Hour.
- **Forecast Horizon**: 24-Hour Day-Ahead horizon (with dynamic 12h, 48h, 72h, and 168h multi-horizon support).
- **Facility Portfolio**: 64 building meters across 3 multi-zone university campuses (Bundoora, Albury-Wodonga, Bendigo).
- **Champion Model**: LightGBM Gradient Boosted Decision Trees achieving an **$R^2$ of 0.9371** and **MAE of 6.33 kWh** on an out-of-sample forward test set.
- **Explainability**: Integrated game-theoretic **TreeSHAP** for global feature attribution, directional beeswarm plots, thermal sensitivity curves, and local prediction waterfall decompositions.

---

## 🏗️ End-to-End Workflow: All 17 Lifecycle Steps

1. **Problem Formulation**: Defined target as hourly electricity consumption ($kWh/h$) with a 24-hour day-ahead operational horizon.
2. **Multi-Source Ingestion**: Ingested and audited 8,095,524 raw 15-minute building readings, 7,396,520 weather observations, calendar schedules, campus infrastructure, and maintenance logs.
3. **Exploratory Data Analysis (EDA)**: Discovered a strong two-tier diurnal profile (noon peak at 12:00 PM, base trough at 02:00 AM), 15.9% weekend setback drop, and summer cooling degree peaks in February.
4. **Data Cleansing**: Rectified 8,493 negative readings (0.105% sensor calibration artifacts), capped statistical anomalies, and normalized hours with missing 15-minute pulses.
5. **Seasonal & Calendar Analysis**: Uncovered HVAC cooling inflection thresholds exceeding 24°C and high base-load ratios in library and mixed-use facilities.
6. **Leakage-Free Feature Engineering**: Created strictly forward-available predictors using $t \ge 24h$ autoregressive lags (`lag_24`, `lag_48`, `lag_168`), shifted rolling windows, cyclical trigonometric encoders, and thermal degree hours.
7. **Chronological Splitting**: Enforced strict forward-in-time partitioning (Train: 2018–mid 2021 [1.68M rows], Validation: late 2021 [233K rows], Test: 2022 [142K rows]). Shuffling was strictly prohibited.
8. **Model Development**: Benchmarked Seasonal Naive baseline, Ridge Regression, Random Forest, XGBoost, and LightGBM.
9. **Time-Series Cross Validation (CV)**: Validated temporal stability across 3 expanding-window folds ($R^2 = 0.9388 \pm 0.019$, $MAE = 6.27 \pm 0.18$ kWh).
10. **Quantitative Evaluation**: Evaluated using MAE, RMSE, $R^2$, and MAPE on unseen future observations.
11. **Trend & Anomaly Interpretation**: Quantified basal overnight floor loads (~40 kWh/h) and midday heatwave surges.
12. **Evidence-Based Energy Optimization**: Formulated 4 strategic energy management pillars (thermal pre-cooling, overnight setback automation, calendar BMS synchronization, and DCV retrofits).
13. **Model & Artifact Persistence**: Serialized fitted boosters, TreeExplainer, scaler, and metadata into compact `models/` artifacts.
14. **Streamlit Intelligence Dashboard**: Developed an interactive dashboard with 6 specialized views (Executive Overview, Live Forecast, EDA, Leaderboard, SHAP Explainability, and Sustainability).
15. **Integration & Automated Testing**: Validated pipelines via `tests/test_pipeline.py` (6/6 tests passing) and `src/audit_pipeline.py`.
16. **Deployment Packaging**: Packaged dependencies in `requirements.txt`, documented configurations, and verified local daemon execution on port 8502.
17. **Documentation & Limitations**: Formulated comprehensive documentation of assumptions, sensor boundaries, and operational sustainability impacts.

---

## 🏆 Model Performance & Evaluation Leaderboard

All models were evaluated on the forward-in-time test set (January 1, 2022 – April 30, 2022, 142,426 records):

| Model Architecture | Val MAE (kWh) | Val RMSE (kWh) | Val $R^2$ | Test MAE (kWh) | Test RMSE (kWh) | Test $R^2$ | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Champion** | **5.3386** | **11.1538** | **0.9422** | **6.3344** | **12.7149** | **0.9371** | **22.32%** |
| **XGBoost Regressor** | 5.2900 | 11.2663 | 0.9411 | 6.3934 | 12.9567 | 0.9347 | 22.15% |
| **Random Forest** | 5.4201 | 11.8615 | 0.9347 | 6.4674 | 13.4899 | 0.9292 | 20.67% |
| **Ridge Regression** | 6.8849 | 12.7843 | 0.9241 | 7.8803 | 14.5850 | 0.9172 | 36.47% |
| **Seasonal Naive Baseline** | 8.3115 | 17.9650 | 0.8501 | 10.4358 | 21.8000 | 0.8151 | 33.47% |

### Metric Descriptions & Operational Value:
- **MAE (Mean Absolute Error)**: Measures average operational forecast deviation ($6.33$ kWh vs average facility load of ~52 kWh, representing a <12% relative error).
- **RMSE (Root Mean Squared Error)**: Heavily penalizes large prediction spikes ($12.71$ kWh), critical for preventing costly peak-demand tariff penalties on campus substations.
- **$R^2$ (Coefficient of Determination)**: Captures 93.7% of total variance across 64 distinct building micro-climates and usage profiles.

---

## 🧠 Explainable AI (TreeSHAP) Insights

Top predictive drivers identified by game-theoretic TreeSHAP feature attribution:
1. `lag_24` (Mean $|SHAP| = 18.68$ kWh): Yesterday's consumption at the exact same hour provides the primary baseline anchor.
2. `rolling_mean_24h_lag24` (Mean $|SHAP| = 7.95$ kWh): Represents the recent thermal and operational equilibrium of the facility.
3. `lag_168` (Mean $|SHAP| = 7.11$ kWh): Identical hour from the previous week, encoding recurring weekly scheduling patterns.
4. `is_business_hours` (Mean $|SHAP| = 1.53$ kWh): Binary indicator of active occupancy and lighting schedules.
5. `air_temperature` & `apparent_temperature`: Non-linear scaling during cooling-degree hours ($>24^\circ$C).

> **Important Distinction**: SHAP attribution quantifies mathematical feature importance inside the model. While features exhibit strong predictive influence, physical causation is governed by building thermodynamic laws and HVAC control loops.

---

## 💡 Evidence-Based Energy Optimization Recommendations

1. **Overnight Base Load Automation**:
   - *Evidence*: Unoccupied nighttime demand (01:00–04:00) averages 38.6 kWh/h (72.8% of midday peak).
   - *Action*: Implement smart scheduling and automated submeter branch disconnects for non-essential plug loads and emergency lighting overrides.
2. **Thermal Pre-Cooling & Peak Shaving**:
   - *Evidence*: Heatwave days ($>30^\circ$C) cause a 14.2% surge in cooling power compared to baseline days.
   - *Action*: Use day-ahead predictions to pre-cool thermal mass during cooler dawn hours (05:00–07:30) before peak grid tariff pricing.
3. **Dynamic Academic Calendar Setback**:
   - *Evidence*: Weekends drop by 15.9%, yet some teaching halls retain active HVAC runtime during semester breaks.
   - *Action*: Synchronize BMS setpoints with academic exam and vacation calendars to expand temperature deadbands ($19^\circ\text{C} - 26^\circ\text{C}$) during vacancies.

---

## 🚀 Installation & Local Execution

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-org/Intelligent-Energy-Forecasting.git
cd Intelligent-Energy-Forecasting

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python -m unittest tests/test_pipeline.py
python src/audit_pipeline.py
```

### 3. Launch the Streamlit Intelligence Platform
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501` (or `http://localhost:8502`).

---

## ☁️ Streamlit Community Cloud Deployment Guide

1. **Repository Footprint**: The repository is optimized at **~35 MB total** (well under GitHub's 100 MB file limit and 2 GB repo limit).
   - `.gitignore` automatically excludes heavy raw datasets (>1 GB CSVs) and offline training parquets while preserving the lightweight serving cache (`outputs/dashboard_data.parquet` [9.98 MB]), production model weights (`models/lightgbm_forecaster.joblib` [3.23 MB]), metadata, and visuals.
2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "feat: EnergiSense AI production deployment package"
   git branch -M main
   git remote add origin https://github.com/<your-username>/Intelligent-Energy-Forecasting.git
   git push -u origin main
   ```
3. **Deploy on Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/) and authorize your GitHub account.
   - Click **New app** and select:
     - **Repository**: `<your-username>/Intelligent-Energy-Forecasting`
     - **Branch**: `main`
     - **Main file path**: `app.py`
   - Click **Deploy!**. Streamlit Cloud will install `requirements.txt` and boot the dashboard in under 30 seconds.

---

## ⚠️ Limitations & Boundary Conditions

- **Tariff Data Absence**: The dataset does not provide actual utility tariff billing structures (e.g. dynamic time-of-use rates or demand ratchets). Monetary savings must not be claimed without contractual utility rate sheets.
- **Weather Projections**: Real-time forward forecasting in production relies on third-party meteorological API weather forecasts (e.g. Open-Meteo / BOM) which carry independent forecast uncertainty.
- **Physical Envelope Metadata**: Several buildings in the metadata table lack gross floor area or built year records; median imputation was utilized to ensure cross-building generalization.
