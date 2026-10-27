import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set clean styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def run_eda():
    print("="*60)
    print("RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    df_path = os.path.join(DATA_DIR, "cleaned_hourly_consumption.parquet")
    df = pd.read_parquet(df_path)
    print(f"Loaded dataset: {df.shape[0]:,} records, {df.shape[1]} columns.")

    # Time-based columns for analysis
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['month_name'] = df['timestamp'].dt.month_name()
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # 1. Overall Time Series Plot (Campus 1 & Key Buildings)
    print("Plotting overall time series trend...")
    fig, ax = plt.subplots(figsize=(14, 5))
    daily_campus = df.groupby(['timestamp', 'campus_name'])['consumption_kwh'].sum().reset_index()
    daily_resample = daily_campus.set_index('timestamp').groupby('campus_name')['consumption_kwh'].resample('D').sum().reset_index()
    for campus in daily_resample['campus_name'].unique():
        sub = daily_resample[daily_resample['campus_name'] == campus]
        ax.plot(sub['timestamp'], sub['consumption_kwh'], label=f"Campus: {campus}", alpha=0.85, lw=1.2)
    ax.set_title("Daily Total Electricity Consumption by Campus (2018 - 2022)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Daily Total Consumption (kWh)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_campus_daily_trend.png"), dpi=200)
    plt.close()

    # 2. Diurnal (Hourly) Consumption Pattern: Weekday vs Weekend
    print("Plotting hourly diurnal patterns...")
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_wk = df.groupby(['hour', 'is_weekend'])['consumption_kwh'].mean().unstack()
    hourly_wk.columns = ['Weekday (Mon-Fri)', 'Weekend (Sat-Sun)']
    hourly_wk.plot(ax=ax, color=['#1f77b4', '#ff7f0e'], lw=2.5, marker='o', markersize=4)
    ax.set_title("Average Diurnal Electricity Demand Profile (Hourly)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Hour of Day (0 - 23)", fontsize=11)
    ax.set_ylabel("Mean Consumption (kWh)", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_diurnal_pattern.png"), dpi=200)
    plt.close()

    # 3. Day of Week Consumption Profile
    print("Plotting day of week profile...")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = df.groupby('day_name')['consumption_kwh'].agg(['mean', 'median', 'std']).reindex(days_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(dow_stats.index, dow_stats['mean'], color='#2ca02c', alpha=0.85, edgecolor='#1b611b')
    ax.set_title("Average Consumption by Day of Week", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Average Consumption (kWh/h)", fontsize=11)
    ax.set_xlabel("Day of Week", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, f"{h:.2f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_day_of_week.png"), dpi=200)
    plt.close()

    # 4. Monthly / Seasonal Profile
    print("Plotting monthly seasonal pattern...")
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    monthly_stats = df.groupby('month_name')['consumption_kwh'].mean().reindex(months_order)
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_stats.plot(kind='bar', ax=ax, color='#9467bd', alpha=0.85, edgecolor='#5c3566')
    ax.set_title("Monthly Average Electricity Consumption (Seasonal Dynamics)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Mean Consumption (kWh/h)", fontsize=11)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_xticklabels(monthly_stats.index, rotation=35, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_monthly_seasonality.png"), dpi=200)
    plt.close()

    # 5. Building Category Comparison
    print("Plotting consumption by building category...")
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_stats = df.groupby('category')['consumption_kwh'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    ax.barh(cat_stats.index, cat_stats['mean'], color='#17becf', edgecolor='#0e717b')
    ax.set_title("Mean Hourly Consumption by Building Category", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Hourly Consumption (kWh)", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_building_categories.png"), dpi=200)
    plt.close()

    # 6. Temperature vs Consumption (Inflection curve / HVAC load)
    print("Plotting temperature vs consumption relationship...")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Bin temperature in 2-degree increments
    df['temp_bin'] = pd.cut(df['air_temperature'], bins=np.arange(0, 46, 2))
    temp_profile = df.groupby('temp_bin', observed=False)['consumption_kwh'].agg(['mean', 'count'])
    # Extract midpoint of bins
    midpoints = [b.mid for b in temp_profile.index]
    ax.plot(midpoints, temp_profile['mean'], marker='o', lw=2.2, color='#d62728', label='Mean Consumption')
    ax.set_title("Air Temperature vs. Building Energy Consumption (Thermal Response)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Air Temperature (°C)", fontsize=11)
    ax.set_ylabel("Mean Hourly Consumption (kWh)", fontsize=11)
    ax.axvline(18.0, color='gray', linestyle='--', alpha=0.7, label='Comfort Baseline (~18°C)')
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_weather_temperature_curve.png"), dpi=200)
    plt.close()

    # Save summary insights
    eda_summary = {
        "hourly_diurnal_peak_hour": int(df.groupby('hour')['consumption_kwh'].mean().idxmax()),
        "hourly_diurnal_trough_hour": int(df.groupby('hour')['consumption_kwh'].mean().idxmin()),
        "weekday_mean_kwh": float(df[df['is_weekend'] == 0]['consumption_kwh'].mean()),
        "weekend_mean_kwh": float(df[df['is_weekend'] == 1]['consumption_kwh'].mean()),
        "weekend_reduction_pct": float(
            (df[df['is_weekend'] == 0]['consumption_kwh'].mean() - df[df['is_weekend'] == 1]['consumption_kwh'].mean())
            / df[df['is_weekend'] == 0]['consumption_kwh'].mean() * 100
        ),
        "category_ranking": cat_stats.to_dict(orient="index"),
        "highest_consumption_month": str(df.groupby('month_name')['consumption_kwh'].mean().idxmax()),
        "lowest_consumption_month": str(df.groupby('month_name')['consumption_kwh'].mean().idxmin())
    }

    with open(os.path.join(OUTPUT_DIR, "eda_summary.json"), "w") as f:
        json.dump(eda_summary, f, indent=2)

    print("EDA completed! Figures and eda_summary.json generated.")

if __name__ == "__main__":
    run_eda()
