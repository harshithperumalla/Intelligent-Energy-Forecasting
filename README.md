# Intelligent Energy Consumption Forecasting for Smart Buildings
### EnergiSense AI — Machine Learning and Explainable AI for Energy Intelligence

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Forecasting-green)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![Explainable AI](https://img.shields.io/badge/Explainable%20AI-SHAP-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 1. Project Overview

**EnergiSense AI** is a machine learning-based energy forecasting and analytics platform designed to help universities and commercial buildings understand, predict, and optimize electricity consumption.

The system analyzes historical electricity usage, weather conditions, time-based patterns, and building operating schedules to forecast future energy demand. An interactive Streamlit dashboard presents consumption trends, forecast results, model comparisons, and explainable AI insights.

The project aims to support data-driven energy management, improve demand planning, identify unusual consumption patterns, and help facility managers investigate potential energy optimization opportunities.

## 2. Problem Statement

Buildings consume electricity throughout the day for lighting, heating, ventilation, air conditioning (HVAC), equipment, and other operations. However, energy consumption varies according to weather, occupancy, operating schedules, and seasonal conditions.

Without reliable forecasting, facility managers may struggle to anticipate demand peaks, identify unnecessary overnight consumption, and plan energy management strategies effectively.

Traditional monitoring systems primarily report historical consumption and may not provide sufficient predictive insights.

**The problem is to develop a data-driven system that forecasts hourly electricity consumption from historical energy and relevant contextual information, evaluates forecasting accuracy, explains the factors influencing predictions, and presents actionable insights through an interactive dashboard.**

## 3. Project Objectives

- Analyze historical electricity consumption across multiple buildings.
- Identify daily, weekly, and seasonal consumption patterns.
- Process missing, invalid, and abnormal energy readings.
- Integrate relevant weather, calendar, and time-based information where available.
- Engineer features suitable for time-series forecasting.
- Train and compare multiple machine learning models.
- Evaluate forecasting performance using appropriate regression metrics.
- Apply SHAP-based explainability to understand important prediction drivers.
- Develop an interactive dashboard for exploring energy data and forecasts.
- Generate potential energy management recommendations supported by data analysis.
## 🌐 Live Project Demo

**EnergiSense AI — Intelligent Energy Consumption Forecasting for Smart Buildings** is deployed using Streamlit Community Cloud.

🔗 **Live Dashboard:** https://intelligent-energy-forecasting-9j4smq7udtbzvqtf4at7p8.streamlit.app/

Explore the interactive dashboard to examine building energy consumption, forecast future electricity demand, compare machine learning models, understand prediction drivers using Explainable AI, and review potential energy optimization opportunities.

### Dashboard Features

- **Executive Overview:** Review key energy consumption indicators and building-level insights.
- **Interactive Forecast:** Explore electricity demand forecasts for selected building meters and forecasting horizons.
- **Exploratory Analytics:** Investigate consumption patterns and temporal trends.
- **Model Benchmarking:** Compare the performance of forecasting models using evaluation metrics.
- **Explainable AI (SHAP):** Examine feature importance and understand factors influencing model predictions.
- **Energy Optimization:** Explore data-driven recommendations for improving building energy management.

### Deployment Technology

- Python
- Streamlit
- LightGBM
- SHAP
- GitHub
- Streamlit Community Cloud

**Note:** Forecasts and optimization recommendations are analytical outputs. Actual energy savings depend on building conditions, implementation, and subsequent measurement.

## 4. Dataset Description

The project uses the **UNICON Smart Building Energy & Utilities dataset**, as identified in the project specification.

The dataset supports the analysis of electricity consumption across multiple building meters and the investigation of relationships between energy usage, time, weather, and operational patterns.

### Data categories

| Data category | Purpose |
|---|---|
| Electricity consumption readings | Define the forecasting target and analyze demand patterns. |
| Timestamp information | Identify hourly, daily, weekly, and seasonal behavior. |
| Weather observations | Investigate the relationship between environmental conditions and electricity demand. |
| Calendar information | Capture weekday, weekend, holiday, and academic schedule effects where available. |
| Building metadata | Support analysis across different facilities and building types where available. |
| Maintenance and operational information | Provide additional context when these records are available and integrated. |

### Data preparation

The data preparation process may include:

1. Inspecting dataset structure, columns, data types, and time coverage.
2. Converting timestamps into a consistent datetime format.
3. Identifying duplicate records and missing observations.
4. Checking negative, invalid, and abnormal electricity readings.
5. Aligning energy and weather data by timestamp and relevant building identifiers.
6. Handling missing values using appropriate, documented methods.
7. Resampling or aggregating readings to the required hourly resolution.
8. Validating the resulting dataset before model training.

The exact preprocessing methods depend on the available columns and the implemented pipeline. Negative readings and statistical outliers should be investigated before removal or modification because some unusual values may represent genuine operating events.

## 5. Proposed System

EnergiSense AI follows an end-to-end machine learning workflow:

**Data Collection → Data Preprocessing → Exploratory Analysis → Feature Engineering → Time-Series Splitting → Model Training → Model Evaluation → Explainable AI → Forecasting Dashboard → Energy Insights**

### System components

- **Data Processing:** Cleans, transforms, and prepares the energy dataset.
- **Exploratory Data Analysis:** Examines consumption patterns, weather relationships, and temporal variations.
- **Feature Engineering:** Generates time-based, lagged, rolling, and weather-related features where appropriate.
- **Forecasting Engine:** Trains and evaluates candidate machine learning models.
- **Explainable AI:** Uses SHAP to interpret the contribution of model features.
- **Interactive Dashboard:** Displays historical trends, predictions, evaluation metrics, and optimization opportunities.

## 6. End-to-End Workflow: 17 Steps

### Phase 1: Problem Understanding and Data Preparation

**Step 1 — Problem Formulation**

Define hourly electricity consumption as the prediction target and establish the forecasting horizon and evaluation objectives.

**Step 2 — Data Collection and Ingestion**

Load the available electricity consumption data and relevant weather, calendar, building, and operational datasets.

**Step 3 — Exploratory Data Analysis**

Examine consumption distributions, peak demand periods, daily patterns, weekday and weekend differences, and relationships with weather conditions.

**Step 4 — Data Cleaning**

Identify missing readings, duplicate records, invalid values, and potential sensor anomalies. Apply documented cleaning and imputation methods.

**Step 5 — Seasonal and Calendar Analysis**

Investigate hourly, daily, weekly, and seasonal patterns to understand how building operations and environmental conditions relate to electricity consumption.

### Phase 2: Feature Engineering and Model Development

**Step 6 — Feature Engineering**

Create relevant predictors, such as:

- Hour of day and day of week
- Weekend and calendar indicators
- Lagged electricity consumption
- Historical rolling statistics
- Temperature-related features
- Cyclical time encodings

Only information available at the forecast creation time should be used. Rolling windows and lag features must be shifted appropriately to prevent future-data leakage.

**Step 7 — Chronological Data Splitting**

Divide the data into training, validation, and test periods in chronological order. This simulates forecasting future observations from historical data.

**Step 8 — Model Development**

Train and compare suitable forecasting models, including:

- Seasonal Naive Baseline
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

The final model selection should be based on measured validation performance and operational suitability.

**Step 9 — Time-Series Cross-Validation**

Evaluate model stability using chronological cross-validation, where each validation period follows its corresponding training period.

**Step 10 — Model Evaluation**

Measure forecasting performance using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Mean Absolute Percentage Error (MAPE), where appropriate

### Phase 3: Interpretation and Deployment

**Step 11 — Consumption Trend Interpretation**

Analyze predicted demand, peak periods, overnight consumption, and differences across available building categories.

**Step 12 — Energy Optimization Insights**

Develop potential recommendations using observed consumption patterns, forecast results, and relevant operating conditions.

**Step 13 — Model Persistence**

Save the trained model, feature metadata, and required serving artifacts so that the application can load them without repeating the full training process.

**Step 14 — Dashboard Development**

Build an interactive Streamlit dashboard to present energy consumption, forecasts, model performance, and explainability results.

**Step 15 — Integration and Testing**

Test data processing, feature generation, model loading, prediction generation, and dashboard functionality.

**Step 16 — Deployment Preparation**

Configure dependencies, repository files, application entry points, and deployment settings for the intended hosting environment.

**Step 17 — Documentation and Sustainability Assessment**

Document the problem, dataset, methodology, evaluation results, assumptions, limitations, and potential applications in sustainable building management.

## 7. Machine Learning Models

The project compares different approaches to understand which model provides the most suitable forecasting performance.

| Model | Purpose |
|---|---|
| Seasonal Naive | Establishes a simple historical-pattern baseline. |
| Ridge Regression | Provides a regularized linear regression benchmark. |
| Random Forest | Captures nonlinear relationships using an ensemble of decision trees. |
| XGBoost | Applies gradient boosting to learn complex feature relationships. |
| LightGBM | Provides an efficient gradient-boosted tree approach for tabular forecasting data. |

### Champion model

LightGBM is identified as the candidate champion model in the project specification.

Its final designation should be supported by reproducible validation results. The selected model should then be evaluated on an untouched chronological test set.

### Evaluation metrics

**MAE:** Measures the average absolute difference between actual and predicted consumption.

**RMSE:** Gives greater weight to large prediction errors.

**R²:** Measures how much variation in the target is explained by the model relative to a mean-prediction baseline.

**MAPE:** Expresses absolute errors as percentages. It can be misleading when actual consumption is zero or close to zero, so it should be interpreted alongside MAE and RMSE.

Model performance should be evaluated on comparable observations and reported with the relevant validation or test period clearly identified.

## 8. Explainable AI Using SHAP

EnergiSense AI incorporates SHAP (SHapley Additive exPlanations) to help explain the predictions produced by tree-based machine learning models.

SHAP assigns feature contributions to individual predictions relative to a model-specific reference value.

### Explainability objectives

- Identify the features that contribute most to model predictions.
- Examine the importance of historical consumption.
- Understand how time-based features influence predicted demand.
- Investigate the relationship between weather conditions and consumption.
- Explain individual predictions through feature contribution analysis.

### Potential explainability visualizations

Depending on the implemented dashboard, the explainability module can include:

- Global feature importance plots
- SHAP summary or beeswarm plots
- Feature dependence plots
- Local prediction waterfall plots

**Important:** SHAP values explain how a model uses its inputs. They do not establish that a feature physically causes a change in electricity consumption.

## 9. Energy Optimization Opportunities

Forecasts and consumption analytics can help facility managers investigate potential improvements in building energy management.

### 1. Overnight Base-Load Analysis

**Objective:** Identify buildings or periods with consistently high overnight electricity consumption.

**Potential action:** Investigate equipment schedules, standby loads, HVAC operating hours, and unnecessary consumption.

### 2. Peak-Demand Planning

**Objective:** Anticipate periods of elevated electricity demand.

**Potential action:** Evaluate scheduling changes, equipment coordination, and load-shifting opportunities where operational requirements permit.

### 3. Weather-Aware Energy Management

**Objective:** Understand how temperature and other available weather variables relate to electricity consumption.

**Potential action:** Investigate HVAC schedules and temperature setpoints while maintaining occupant comfort and safety requirements.

### 4. Calendar-Based Scheduling

**Objective:** Compare energy consumption during weekdays, weekends, holidays, and academic breaks where calendar information is available.

**Potential action:** Align building operating schedules with actual occupancy and planned activities.

These are potential optimization strategies, not guaranteed energy savings. Actual savings and emissions reductions require suitable baselines, verified implementation, and post-intervention measurement.

## 10. EnergiSense AI Dashboard

The Streamlit dashboard is intended to make forecasting results accessible to users who may not work directly with machine learning code.

### Dashboard modules

| Module | Purpose |
|---|---|
| Executive Overview | Summarizes energy consumption, key indicators, and major trends. |
| Interactive Forecast | Allows users to explore available forecasting outputs. |
| Exploratory Analytics | Displays consumption patterns and relationships in the data. |
| Model Benchmarking | Compares forecasting models using evaluation metrics. |
| Explainable AI (SHAP) | Presents feature importance and prediction explanations. |
| Energy Optimization | Displays potential energy management insights and recommendations. |

The final README should describe only the pages and interactive features implemented in the current application.

## 11. Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization, where used |
| Scikit-learn | Preprocessing, baseline models, and evaluation |
| LightGBM | Gradient-boosted tree forecasting |
| XGBoost | Alternative gradient-boosting model |
| SHAP | Model explainability |
| Streamlit | Interactive web dashboard |
| Joblib | Model and artifact persistence, where used |
| Git and GitHub | Version control and source code management |

The installed versions and actual dependencies should be taken from the project's `requirements.txt`.

## 12. Project Structure

The following is an illustrative structure. Adjust it to match the actual repository.

```text
Intelligent-Energy-Forecasting/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── README.md
│
├── models/
│   └── lightgbm_forecaster.joblib
│
├── outputs/
│   └── dashboard_data.parquet
│
├── src/
│   └── audit_pipeline.py
│
├── tests/
│   └── test_pipeline.py
│
└── assets/
    └── dashboard visualizations
```

Large raw datasets and training artifacts may be excluded from version control. If they are required for training or serving, document how to obtain or generate them.

## 13. Installation and Local Execution

### Prerequisites

- Python version compatible with the project dependencies
- Git
- Required dataset files or prepared serving artifacts

### Step 1: Clone the repository

```bash
git clone https://github.com/harshithperumalla/Intelligent-Energy-Forecasting.git
cd Intelligent-Energy-Forecasting
```

### Step 2: Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
python -m streamlit run app.py
```

Open the local address displayed in the terminal. Streamlit commonly uses port `8501` by default, but the actual port may differ if it is already occupied or explicitly configured.

### Step 5: Run tests

If the repository includes the stated test and audit scripts, run:

```bash
python -m unittest tests/test_pipeline.py
python src/audit_pipeline.py
```

Successful execution depends on the test configuration, dependencies, and availability of any required input files.

## 14. Deployment

The application can be prepared for deployment using Streamlit Community Cloud or another compatible hosting platform.

For Streamlit Community Cloud:

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect the relevant GitHub account.
4. Create a new application.
5. Select the repository and branch.
6. Set `app.py` as the main application file.
7. Deploy and inspect the application logs.

Before deployment, verify that:

- `requirements.txt` contains compatible dependencies.
- All required model artifacts are included or generated during startup.
- The application does not depend on local files excluded from Git.
- File paths work in the cloud environment.
- The application can start without requiring the full training dataset, unless that is intentional.

**Deployment status:** Update this section with the public application URL only after a successful deployment has been confirmed.

## 15. Limitations

- **Data quality:** Missing observations, meter faults, and measurement errors can affect model reliability.
- **Forecast uncertainty:** Prediction accuracy may change across buildings, seasons, and unusual operating conditions.
- **Weather uncertainty:** Forecasts that rely on future weather data are subject to the uncertainty of the weather forecast source.
- **Building differences:** Variations in occupancy, floor area, equipment, and operating schedules can affect generalization.
- **Tariff limitations:** Financial savings cannot be reliably calculated without applicable electricity tariffs and demand-charge structures.
- **Explainability limitations:** Feature attribution describes model behavior but does not prove physical causation.
- **Operational validation:** Recommendations must be assessed against building-specific constraints before implementation.
- **Sustainability measurement:** Actual energy savings and emissions reductions require appropriate baseline comparisons and post-implementation measurements.

## 16. Expected Outcomes

The project is designed to provide:

- Forecasts of hourly electricity consumption.
- Better visibility into daily, weekly, and seasonal demand patterns.
- Quantitative comparisons of candidate forecasting models.
- Explainable predictions using feature attribution.
- Identification of periods that may warrant further energy investigation.
- An interactive interface for reviewing energy data and model outputs.
- A foundation for future integration with building management systems and operational optimization tools.

These are intended outcomes. Their successful achievement should be demonstrated through the implemented application and its evaluation results.

## 17. Future Enhancements

Potential future improvements include:

- Integration with live smart-meter data.
- Automated weather forecast ingestion.
- Building-specific forecasting models.
- Uncertainty intervals for future predictions.
- Forecast monitoring and model drift detection.
- Integration with building management systems.
- Tariff-aware demand optimization.
- Measurement and verification of energy-saving interventions.
- Carbon-emissions estimation using suitable emissions factors.

## 18. Conclusion

EnergiSense AI combines time-series analysis, machine learning, explainable AI, and interactive visualization to support intelligent electricity consumption forecasting for smart buildings.

By transforming historical energy data into forecasts and interpretable insights, the project provides a foundation for more informed energy planning and building management.

The system's practical value ultimately depends on reliable data, leakage-free evaluation, reproducible model performance, and validation of proposed interventions in real operating environments.

## License

This project is distributed under the MIT License if the repository includes the corresponding MIT license terms. See the repository's `LICENSE` file for the applicable conditions.
