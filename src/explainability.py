import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_shap_analysis():
    print("="*60)
    print("STARTING SHAP EXPLAINABILITY PIPELINE")
    print("="*60)
    t0 = time.time()

    # Load model and feature metadata
    print("Loading LightGBM Champion model and metadata...")
    model_path = os.path.join(MODELS_DIR, "lightgbm_forecaster.joblib")
    lgb_model = joblib.load(model_path)

    meta_path = os.path.join(MODELS_DIR, "feature_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta['feature_cols']

    # Load test dataset sample for SHAP explanation
    print("Loading test dataset for explanation sample...")
    df_path = os.path.join(DATA_DIR, "featured_hourly_consumption.parquet")
    df = pd.read_parquet(df_path)
    test_df = df[df['timestamp'] >= '2022-01-01'].copy()
    print(f"Total test rows: {len(test_df):,}")

    # Select representative sample (e.g. 2,500 samples across buildings and hours)
    # Computationally practical for fast, exact TreeExplainer
    sample_size = min(2500, len(test_df))
    sample_df = test_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    X_sample = sample_df[feature_cols]

    print(f"Computing SHAP values for {sample_size} representative instances using TreeExplainer...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer(X_sample)

    # 1. Global Feature Importance (Mean Absolute SHAP value)
    print("Calculating global feature importance ranking...")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    importance_df.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)
    print("\nTop 10 Most Influential Features:")
    print(importance_df.head(10).to_string(index=False))

    # 2. Save SHAP Summary Bar Chart
    print("\nGenerating SHAP summary bar chart...")
    fig, ax = plt.subplots(figsize=(10, 7))
    top_15 = importance_df.head(15).iloc[::-1]
    ax.barh(top_15['feature'], top_15['mean_abs_shap'], color='#1f77b4', edgecolor='#0d47a1')
    ax.set_title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Absolute SHAP Impact on Predicted Consumption (kWh)", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "shap_global_bar.png"), dpi=200)
    plt.close()

    # 3. Save SHAP Beeswarm Plot (Directional impact: positive/negative contribution)
    print("Generating SHAP beeswarm summary plot...")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot: Feature Impact Directionality", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm_summary.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 4. Temperature Dependence Plot
    print("Generating SHAP temperature dependence plot...")
    plt.figure(figsize=(9, 5.5))
    shap.plots.scatter(shap_values[:, "air_temperature"], color=shap_values[:, "cooling_degree_hours"], show=False)
    plt.title("SHAP Dependence: Air Temperature vs. Energy Impact (kWh)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_temp_dependence.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 5. Local Prediction Explanations for High, Medium, Low consumption instances
    print("Generating local prediction waterfall explanations...")
    high_idx = int(np.argmax(shap_values.values.sum(axis=1)))
    low_idx = int(np.argmin(shap_values.values.sum(axis=1)))
    med_idx = int(np.argsort(shap_values.values.sum(axis=1))[len(shap_values)//2])

    cases = [('high_consumption', high_idx), ('median_consumption', med_idx), ('low_consumption', low_idx)]
    for case_name, idx in cases:
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
        plt.title(f"Local Explanation (Waterfall): {case_name.replace('_', ' ').title()}", fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_waterfall_{case_name}.png"), dpi=200, bbox_inches='tight')
        plt.close()

    # Save explainer object and sample data for interactive use in Streamlit
    print("\nSaving explainer and sample artifact for interactive Streamlit dashboard...")
    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.joblib"))
    sample_df.head(200).to_parquet(os.path.join(OUTPUT_DIR, "shap_interactive_sample.parquet"), index=False)

    # Save summary insights json
    shap_summary = {
        'top_5_features': importance_df['feature'].head(5).tolist(),
        'top_feature_impacts': importance_df['mean_abs_shap'].head(5).tolist(),
        'explanation_method': 'TreeSHAP (TreeExplainer)',
        'base_value_expected_kwh': float(explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value),
        'sample_instances_analyzed': sample_size,
        'elapsed_seconds': round(time.time() - t0, 2)
    }

    with open(os.path.join(OUTPUT_DIR, "shap_summary.json"), "w") as f:
        json.dump(shap_summary, f, indent=2)

    print(f"SHAP analysis completed in {time.time() - t0:.2f}s!")
    return importance_df

if __name__ == "__main__":
    run_shap_analysis()
