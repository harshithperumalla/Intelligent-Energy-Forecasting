import os
import json
import time
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_clean_data(save_intermediate=True):
    print("="*60)
    print("STARTING DATA INGESTION AND CLEANING PIPELINE")
    print("="*60)
    t0 = time.time()
    cleaning_log = {}

    # 1. Load Building Metadata
    print("\n[Step 1/6] Loading building and campus metadata...")
    b_meta_path = os.path.join(DATA_DIR, "building_meta.csv")
    b_meta = pd.read_csv(b_meta_path)
    b_meta = b_meta.rename(columns={'id': 'meter_id'})
    print(f"Loaded building_meta: {b_meta.shape[0]} buildings.")

    c_meta_path = os.path.join(DATA_DIR, "campus_meta.csv")
    c_meta = pd.read_csv(c_meta_path)
    c_meta = c_meta.rename(columns={'id': 'campus_id', 'name': 'campus_name', 'capacity': 'campus_capacity'})
    print(f"Loaded campus_meta: {c_meta.shape[0]} campuses.")

    # Merge campus meta into building meta
    meta_df = pd.merge(b_meta, c_meta, on='campus_id', how='left')

    # Fill missing values in metadata cleanly
    meta_df['category'] = meta_df['category'].fillna('other')
    meta_df['gross_floor_area'] = meta_df['gross_floor_area'].fillna(meta_df['gross_floor_area'].median())
    meta_df['built_year'] = meta_df['built_year'].fillna(meta_df['built_year'].median())
    meta_df['capacity'] = meta_df['capacity'].fillna(meta_df['capacity'].median())

    # 2. Load and Aggregate Calendar Data
    print("\n[Step 2/6] Loading calendar and events...")
    cal_path = os.path.join(DATA_DIR, "calender.csv")
    cal_df = pd.read_csv(cal_path)
    cal_df['date'] = pd.to_datetime(cal_df['date']).dt.date
    # Clean calendar flags
    cal_df['is_holiday'] = cal_df['is_holiday'].fillna(0).astype(int)
    cal_df['is_semester'] = cal_df['is_semester'].fillna(0).astype(int)
    cal_df['is_exam'] = cal_df['is_exam'].fillna(0).astype(int)
    print(f"Loaded calendar: {len(cal_df)} dates.")

    # Events data
    events_path = os.path.join(DATA_DIR, "events.csv")
    has_events = False
    events_daily = None
    if os.path.exists(events_path):
        events_df = pd.read_csv(events_path)
        events_df['date'] = pd.to_datetime(events_df['date']).dt.date
        events_daily = events_df.groupby(['meter_id', 'date']).size().reset_index(name='event_count')
        has_events = True
        print(f"Loaded events: {len(events_df)} event records mapped.")

    # 3. Process Weather Data to Hourly
    print("\n[Step 3/6] Resampling weather data to hourly means...")
    w_cols = ['campus_id', 'timestamp', 'air_temperature', 'apparent_temperature', 'relative_humidity', 'wind_speed']
    weather_chunks = []
    w_rows = 0
    w_path = os.path.join(DATA_DIR, "weather_data.csv")
    for chunk in pd.read_csv(w_path, usecols=w_cols, chunksize=1000000):
        w_rows += len(chunk)
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        chunk['timestamp'] = chunk['timestamp'].dt.floor('h')
        agg = chunk.groupby(['campus_id', 'timestamp']).mean().reset_index()
        weather_chunks.append(agg)

    df_weather = pd.concat(weather_chunks, ignore_index=True)
    df_weather_hourly = df_weather.groupby(['campus_id', 'timestamp']).mean().reset_index()
    # Interpolate short weather missing values
    df_weather_hourly = df_weather_hourly.sort_values(['campus_id', 'timestamp']).reset_index(drop=True)
    for col in ['air_temperature', 'apparent_temperature', 'relative_humidity', 'wind_speed']:
        df_weather_hourly[col] = df_weather_hourly.groupby('campus_id')[col].transform(lambda x: x.ffill().bfill())

    print(f"Processed {w_rows:,} raw weather rows into {len(df_weather_hourly):,} hourly weather records.")

    # 4. Ingest and Clean Building Consumption (15-min -> 1-hour)
    print("\n[Step 4/6] Processing and cleaning building consumption...")
    c_path = os.path.join(DATA_DIR, "building_consumption.csv")
    total_raw_rows = 0
    negative_readings_count = 0
    extreme_outliers_count = 0
    hourly_chunks = []

    for chunk in pd.read_csv(c_path, chunksize=1000000):
        total_raw_rows += len(chunk)
        
        # Check negative readings
        neg_mask = chunk['consumption'] < 0
        negative_readings_count += neg_mask.sum()
        # Clean: set negative consumption to 0.0 (sensor artifact / back-feed)
        chunk.loc[neg_mask, 'consumption'] = 0.0
        
        # Check extreme spikes (> 99.999% threshold / physically implausible single 15-min reading > 500 kWh)
        spike_mask = chunk['consumption'] > 500.0
        extreme_outliers_count += spike_mask.sum()
        chunk.loc[spike_mask, 'consumption'] = 500.0

        # Resample to hourly: timestamp to floor hour
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        chunk['timestamp'] = chunk['timestamp'].dt.floor('h')
        
        # Aggregate sum of consumption and count of readings per hour
        agg = chunk.groupby(['campus_id', 'meter_id', 'timestamp']).agg(
            consumption_kwh=('consumption', 'sum'),
            reading_count=('consumption', 'count')
        ).reset_index()
        
        hourly_chunks.append(agg)

    print(f"Aggregating hourly consumption across chunks...")
    df_hourly = pd.concat(hourly_chunks, ignore_index=True)
    df_hourly = df_hourly.groupby(['campus_id', 'meter_id', 'timestamp']).agg(
        consumption_kwh=('consumption_kwh', 'sum'),
        reading_count=('reading_count', 'sum')
    ).reset_index()

    # Normalize partial hours (e.g. if 3 of 4 readings were present, normalize to 4 readings)
    df_hourly['normalized_kwh'] = np.where(
        (df_hourly['reading_count'] >= 2) & (df_hourly['reading_count'] < 4),
        df_hourly['consumption_kwh'] * (4.0 / df_hourly['reading_count']),
        df_hourly['consumption_kwh']
    )
    df_hourly['consumption_kwh'] = df_hourly['normalized_kwh'].round(4)
    df_hourly = df_hourly.drop(columns=['normalized_kwh', 'reading_count'])

    cleaning_log["total_raw_readings"] = int(total_raw_rows)
    cleaning_log["negative_readings_cleaned"] = int(negative_readings_count)
    cleaning_log["extreme_spikes_capped"] = int(extreme_outliers_count)
    cleaning_log["total_hourly_records"] = int(len(df_hourly))

    print(f"Total raw 15-min readings: {total_raw_rows:,}")
    print(f"Negative readings cleaned: {negative_readings_count:,} ({negative_readings_count/total_raw_rows*100:.3f}%)")
    print(f"Extreme outliers capped: {extreme_outliers_count:,}")
    print(f"Total clean hourly records: {len(df_hourly):,}")

    # 5. Join Weather, Calendar, Metadata
    print("\n[Step 5/6] Merging weather, calendar, and building metadata...")
    # Merge weather on [campus_id, timestamp]
    merged = pd.merge(df_hourly, df_weather_hourly, on=['campus_id', 'timestamp'], how='left')

    # Merge building metadata on [meter_id]
    merged = pd.merge(merged, meta_df, on=['meter_id', 'campus_id'], how='left')

    # Add date column for calendar join
    merged['date'] = merged['timestamp'].dt.date
    merged = pd.merge(merged, cal_df, on='date', how='left')

    # Merge events if available
    if has_events and events_daily is not None:
        merged = pd.merge(merged, events_daily, on=['meter_id', 'date'], how='left')
        merged['event_count'] = merged['event_count'].fillna(0).astype(int)
    else:
        merged['event_count'] = 0

    # Fill weather missing values if any
    for col in ['air_temperature', 'apparent_temperature', 'relative_humidity', 'wind_speed']:
        merged[col] = merged.groupby('campus_id')[col].transform(lambda x: x.ffill().bfill())

    # Drop temporary date
    merged = merged.drop(columns=['date'])

    # Sort chronologically by meter and timestamp
    merged = merged.sort_values(['campus_id', 'meter_id', 'timestamp']).reset_index(drop=True)

    print(f"\nFinal merged hourly dataset shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
    print(f"Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")

    # 6. Save Cleaned Dataset
    print("\n[Step 6/6] Saving cleaned dataset and metadata artifacts...")
    processed_path = os.path.join(DATA_DIR, "cleaned_hourly_consumption.parquet")
    merged.to_parquet(processed_path, index=False)
    print(f"Saved to {processed_path} ({os.path.getsize(processed_path)/(1024*1024):.2f} MB)")

    # Save summary statistics
    stats = {
        "total_records": len(merged),
        "total_buildings": int(merged['meter_id'].nunique()),
        "total_campuses": int(merged['campus_id'].nunique()),
        "min_timestamp": str(merged['timestamp'].min()),
        "max_timestamp": str(merged['timestamp'].max()),
        "consumption_mean": float(merged['consumption_kwh'].mean()),
        "consumption_std": float(merged['consumption_kwh'].std()),
        "consumption_min": float(merged['consumption_kwh'].min()),
        "consumption_max": float(merged['consumption_kwh'].max()),
        "consumption_median": float(merged['consumption_kwh'].median()),
        "cleaning_log": cleaning_log,
        "elapsed_seconds": round(time.time() - t0, 2)
    }

    with open(os.path.join(OUTPUT_DIR, "cleaning_log.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("Cleaning log saved to outputs/cleaning_log.json")
    print("Data processing pipeline complete!")
    return merged

if __name__ == "__main__":
    process_and_clean_data()
