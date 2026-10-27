import os
import time
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_features(df=None, sample_buildings=None):
    """
    Creates leakage-free features for energy consumption forecasting.
    Lags are structured so that all features for hour (t) use information
    from at least 24 hours prior (t - 24, t - 48, t - 168), enabling
    a full 24-hour ahead day-ahead forecast horizon without target leakage.
    """
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    t0 = time.time()

    if df is None:
        parquet_path = os.path.join(DATA_DIR, "cleaned_hourly_consumption.parquet")
        print(f"Loading cleaned dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

    if sample_buildings is not None:
        print(f"Filtering to specific buildings: {sample_buildings}...")
        df = df[df['meter_id'].isin(sample_buildings)].copy()

    # Sort chronologically by meter and timestamp
    df = df.sort_values(['campus_id', 'meter_id', 'timestamp']).reset_index(drop=True)

    print(f"Processing {len(df):,} records...")

    # 1. Temporal & Calendar Features
    print("Creating calendar and cyclical time features...")
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    # Business operating hours indicator (typical university/office hours 08:00 - 18:00 on weekdays)
    df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['is_weekend'] == 0)).astype(int)

    # 2. Weather & Thermal Features
    print("Engineering weather and thermal degree hour features...")
    # Base temperature for comfort baseline = 18.0 C
    df['cooling_degree_hours'] = np.maximum(0.0, df['air_temperature'] - 18.0)
    df['heating_degree_hours'] = np.maximum(0.0, 18.0 - df['air_temperature'])
    df['temp_humidity_index'] = df['air_temperature'] - (0.55 - 0.0055 * df['relative_humidity']) * (df['air_temperature'] - 14.5)

    # 3. Autoregressive Lags (Strictly t - 24 and older for 24-hr day-ahead horizon)
    print("Generating leakage-free autoregressive lags and rolling statistics...")
    # Group by meter_id so lags don't cross between buildings
    grouped = df.groupby('meter_id')['consumption_kwh']

    # Lag 24: same hour yesterday
    df['lag_24'] = grouped.shift(24)
    # Lag 48: same hour 2 days ago
    df['lag_48'] = grouped.shift(48)
    # Lag 168: same hour same day last week (7 days ago * 24h)
    df['lag_168'] = grouped.shift(168)

    # Rolling statistics based on shifted series (shifted by 24h to avoid target leakage)
    shifted_target = grouped.shift(24)
    df['rolling_mean_24h_lag24'] = shifted_target.groupby(df['meter_id']).transform(lambda x: x.rolling(24, min_periods=6).mean())
    df['rolling_std_24h_lag24'] = shifted_target.groupby(df['meter_id']).transform(lambda x: x.rolling(24, min_periods=6).std()).fillna(0)
    df['rolling_max_24h_lag24'] = shifted_target.groupby(df['meter_id']).transform(lambda x: x.rolling(24, min_periods=6).max())
    df['rolling_min_24h_lag24'] = shifted_target.groupby(df['meter_id']).transform(lambda x: x.rolling(24, min_periods=6).min())

    # 4. Building category categorical encoding
    df['category_code'] = df['category'].astype('category').cat.codes

    # 5. Drop initial warm-up period where lag_168 is NaN (first 7 days)
    initial_len = len(df)
    df = df.dropna(subset=['lag_168', 'rolling_mean_24h_lag24']).reset_index(drop=True)
    print(f"Dropped {initial_len - len(df):,} warm-up rows (first 168 hours of each meter).")
    print(f"Remaining clean rows: {len(df):,}")

    # Save to parquet
    save_path = os.path.join(DATA_DIR, "featured_hourly_consumption.parquet")
    df.to_parquet(save_path, index=False)
    print(f"Features saved to {save_path} ({os.path.getsize(save_path)/(1024*1024):.2f} MB)")
    print(f"Feature engineering completed in {time.time() - t0:.2f}s.")
    return df

if __name__ == "__main__":
    create_features()
