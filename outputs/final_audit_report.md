# Final Senior ML Engineer Audit Report
**Project Title**: Intelligent Energy Consumption Forecasting for Smart Buildings  
**Dataset**: UNICON Building Energy & Utilities Dataset  
**Champion Architecture**: LightGBM Gradient Boosted Decision Trees  
**Application**: Streamlit Intelligence Platform (`app.py`, active on port 8502)  
**Date of Audit**: 2026-10-25 16:16:47  

---

## 1. Dataset Files & Usage Audit

| Dataset Filename | Usage Status | Contribution & Role in System | Join Key / Timestamp Verification |
| :--- | :---: | :--- | :--- |
| `building_consumption.csv` | **USED** | **Primary Forecasting Target**: 8,095,524 raw 15-minute readings resampled to hourly electricity demand ($kWh/h$) across 64 buildings (2018–2022). | Master reference table: `[campus_id, meter_id, timestamp]`. |
| `building_meta.csv` | **USED** | **Building Covariates**: 64 facilities providing category (library, office, teaching, mixed use), gross floor area, built year, and room capacity. | Joined 1:1 on `[meter_id == id]`. Verified zero row duplication. |
| `campus_meta.csv` | **USED** | **Campus Spatial Hierarchy**: Supplies campus names (Bundoora, Albury-Wodonga, Bendigo) and campus total design capacities. | Joined 1:1 on `[campus_id == id]`. Verified zero row duplication. |
| `calender.csv` | **USED** | **Academic & Civic Calendar**: Daily flags for public holidays, active academic semesters, and exam periods. | Joined 1:1 on `[timestamp.date == date]`. Verified zero row duplication. |
| `events.csv` | **USED** | **Operational Maintenance**: Facility events (HVAC tuning, AHU decommissioning, scheduled maintenance). | Left joined on `[meter_id, timestamp.date]`. Aggregated to daily counts. |
| `weather_data.csv` | **USED** | **Meteorological Drivers**: 7,396,520 rows resampled to hourly means per campus (air temp, apparent temp, humidity, wind speed, cooling/heating degree hours). | Joined 1:1 on `[campus_id, timestamp.floor('h')]`. Clean interpolation. |
| `gas_consumption.csv` | **OPTIONAL** | **Separate Utility (Gas)**: Measured in gas volume/thermal units. Excluded from electricity model to prevent multi-utility target contamination. | N/A (Preserved as standalone utility record). |
| `water_consumption.csv` | **OPTIONAL** | **Separate Utility (Water)**: Monitored in kL. Excluded to maintain strictly clean electricity forecasting ($kWh$). | N/A (Preserved as standalone utility record). |
| `nmi_consumption.csv` | **OPTIONAL** | **Grid Boundary Connection**: High-voltage campus boundary meters (`demand_kW`, `demand_kVA`). Kept separate from building meters. | N/A (Preserved as standalone utility record). |
| `nmi_meta.csv` | **OPTIONAL** | **Grid Boundary Limits**: Peak demand capacity for NMI connection points. | N/A (Preserved as standalone utility record). |
| `building_submeter_consumption.csv` | **OPTIONAL** | **Sub-circuit Telemetry**: Short 6-month slice for single building (Building 14); `building_consumption.csv` provides complete 4-year building-level record. | N/A (Preserved as standalone utility record). |

---

## 2. All 17 HCL Workflow Requirements Verification

| Step | Requirement | Status | Verification & Evidence |
| :---: | :--- | :---: | :--- |
| **1** | Problem Definition | **PASS** | Target: Hourly electricity demand ($kWh/h$). Horizon: 24h day-ahead (and 12h–168h multi-horizon). Interval: 1h. |
| **2** | Dataset Inspection | **PASS** | Audited 8.09M electricity rows, 7.4M weather rows, 64 buildings, 5 campuses. Saved in `outputs/data_audit_summary.json`. |
| **3** | Consumption Pattern Analysis | **PASS** | Generated diurnal solar curve, weekly profiles, and campus totals. Saved in `outputs/eda_*.png`. |
| **4** | Data Cleansing | **PASS** | 8,493 negative readings (0.105%) cleaned to 0.0; partial hours scaled; weather ffill/bfill; logged in `outputs/cleaning_log.json`. |
| **5** | Seasonal & Temporal Dynamics | **PASS** | Diurnal peak at 12:00 PM, trough at 02:00 AM, 15.9% weekend setback drop, and summer cooling degree surge ($>24^\circ$C). |
| **6** | Leakage-Free Features | **PASS** | Exclusively forward-available predictors ($t \ge 24	ext{h}$ lags: `lag_24`, `lag_48`, `lag_168`, shifted rolling mean/std/min/max, cyclical sine/cosine). |
| **7** | Chronological Split | **PASS** | Forward-only split: Train (< 2021-07-01 [1.68M]), Val (late 2021 [233K]), Test (Jan–Apr 2022 [142K]). Zero random shuffling. |
| **8** | Model Development | **PASS** | Trained Seasonal Naive baseline, Ridge Regression, Random Forest, XGBoost, and Champion LightGBM. |
| **9** | Forecast Validation (CV) | **PASS** | 3-Fold Time-Series Cross Validation verified: Mean $R^2 = 0.9388 \pm 0.019$, Mean $	ext{MAE} = 6.27 \pm 0.18$ kWh. |
| **10** | Metric Evaluation | **PASS** | Comprehensive comparison across MAE, RMSE, $R^2$, and MAPE on out-of-sample forward test set. |
| **11** | Trend & Anomaly Interpretation | **PASS** | Discovered overnight base load ratio (72.8% of midday peak) and non-linear summer cooling degree penalties. |
| **12** | Energy Optimization Insights | **PASS** | 4 actionable pillars formulated without fabricated cost tariffs. Saved in `outputs/energy_insights.json`. |
| **13** | Artifact Persistence | **PASS** | `lightgbm_forecaster.joblib`, `scaler.joblib`, `shap_explainer.joblib`, and `feature_metadata.json` saved and validated. |
| **14** | Streamlit UI Dashboard | **PASS** | Responsive glassmorphic dashboard with 6 specialized views active on port 8502. |
| **15** | Automated Testing | **PASS** | 6/6 unit tests passing (`tests/test_pipeline.py`) covering forecaster, horizons, error handling, and confidence bounds. |
| **16** | Deployment Packaging | **PASS** | `requirements.txt`, `.gitignore`, `README.md`, and lightweight cached test dataset (9.57 MB) fully configured. |
| **17** | Documentation & Sustainability | **PASS** | Detailed documentation of workflow steps, carbon avoidance estimates (~150k–220k kWh annual basal reduction), and limitations. |

---

## 3. Actual Out-of-Sample Model Evaluation Leaderboard

Evaluated on 142,426 out-of-sample chronological test records (January 1, 2022 to April 30, 2022):

| Model | Val_MAE | Val_RMSE | Val_R2 | Val_MAPE_% | Test_MAE | Test_RMSE | Test_R2 | Test_MAPE_% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM_Champion** | 5.3386 | 11.1538 | 0.9422 | 20.31 | 6.3344 | 12.7149 | 0.9371 | 22.32 |
| XGBoost | 5.29 | 11.2663 | 0.9411 | 19.89 | 6.3934 | 12.9567 | 0.9347 | 22.15 |
| Random_Forest | 5.4201 | 11.8615 | 0.9347 | 20.67 | 6.4674 | 13.4899 | 0.9292 | 22.51 |
| Ridge_Regression | 6.8849 | 12.7843 | 0.9241 | 35.09 | 7.8803 | 14.585 | 0.9172 | 36.47 |
| Baseline_Seasonal_Naive | 8.3115 | 17.965 | 0.8501 | 30.77 | 10.4358 | 21.8 | 0.8151 | 33.47 |

---

## 4. SHAP Explainability & Inference Test Results

- **SHAP Engine**: `shap.TreeExplainer` on fitted LightGBM Booster (`models/shap_explainer.joblib`).
- **Global Feature Attribution (Mean |SHAP Value|)**:
  1. `lag_24`: **18.683 kWh** — Primary baseline anchor (consumption yesterday at identical hour).
  2. `rolling_mean_24h_lag24`: **7.955 kWh** — Recent thermal and operational equilibrium.
  3. `lag_168`: **7.112 kWh** — Same hour same day last week (recurring weekly schedule).
  4. `lag_48`: **2.216 kWh** — Two-day historical lag.
  5. `is_business_hours`: **1.533 kWh** — Binary occupancy and lighting trigger.
- **Directional Beeswarm & Thermal Dependence**: Verified non-linear cooling surge above 24°C; saved in `outputs/shap_beeswarm_summary.png` and `outputs/shap_temp_dependence.png`.
- **Inference Stability Test**:
  - 24-Hour Horizon: Successfully generated 24 hourly steps (Mean: 38.46 kWh) with valid 95% confidence intervals ($\pm 1.96 	imes 	ext{RMSE}$).
  - 168-Hour Horizon: Successfully generated 168 hourly steps (Mean: 55.59 kWh).

---

## 5. Issues Fixed & Confirmed System Status

1. **Streamlit Deprecation Warning**: Replaced all legacy `use_column_width=True` calls with `use_container_width=True` across `app.py`.
2. **Import Path Resolution**: Ensured relative path resolution via `BASE_DIR` across all inference and test modules.
3. **Port Conflict**: Gracefully resolved port 8501 contention by configuring local server to port 8502 (`http://localhost:8502`).
4. **Cache Optimization**: Generated `outputs/dashboard_data.parquet` (9.57 MB) to enable instant sub-second dashboard startup without reloading 263 MB raw files.
5. **Remaining Issues**: **None**. All pipelines, tests, and user interface features run with zero unresolved errors.

---

## 6. Streamlit Community Cloud Deployment Instructions

1. **Push Codebase to GitHub**:
   ```bash
   git init
   git add app.py requirements.txt README.md .gitignore models/ outputs/ src/ tests/
   git commit -m "Deploy: Intelligent Energy Consumption Forecasting Platform"
   git branch -M main
   git remote add origin https://github.com/<your-username>/Intelligent-Energy-Forecasting.git
   git push -u origin main
   ```
   *(Note: `.gitignore` automatically excludes raw 263MB CSVs to maintain repository size below GitHub limits, while including precomputed model weights and cached dashboard parquet).*

2. **Connect Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/).
   - Click **New app** and authorize your GitHub account.
   - Select your repository: `<your-username>/Intelligent-Energy-Forecasting`.
   - Branch: `main`.
   - Main file path: `app.py`.
   - Click **Deploy!**.

3. **Deployment Verification**:
   - Streamlit Cloud will install packages from `requirements.txt`.
   - The application will initialize in ~30 seconds using the precomputed models and cached test telemetry.

---
*Audit Completed Successfully by Senior ML Engineer. All 17 Requirements Verified PASS.*
