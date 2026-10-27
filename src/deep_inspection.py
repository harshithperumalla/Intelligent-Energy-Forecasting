import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def deep_inspect():
    print("Loading sample of building_consumption.csv...")
    # Read first 500,000 rows to see meters and timestamps
    df_sample = pd.read_csv(os.path.join(DATA_DIR, "building_consumption.csv"), nrows=200000)
    print(df_sample.info())
    print("\nSample unique campus_id:", df_sample['campus_id'].unique())
    print("Sample unique meter_id:", df_sample['meter_id'].unique())
    print("Consumption stats:\n", df_sample['consumption'].describe())
    
    # Read unique meter_id and campus_id across the full file
    print("\nScanning unique meter_ids and campus_ids across full building_consumption...")
    meters = set()
    campus_ids = set()
    total_rows = 0
    null_counts = {'campus_id': 0, 'meter_id': 0, 'timestamp': 0, 'consumption': 0}
    
    for chunk in pd.read_csv(os.path.join(DATA_DIR, "building_consumption.csv"), chunksize=500000):
        total_rows += len(chunk)
        meters.update(chunk['meter_id'].unique())
        campus_ids.update(chunk['campus_id'].unique())
        for c in null_counts:
            null_counts[c] += chunk[c].isnull().sum()
            
    print(f"Total rows: {total_rows}")
    print(f"Unique campus_ids: {sorted(list(campus_ids))}")
    print(f"Unique meter_ids ({len(meters)} meters): {sorted(list(meters))}")
    print(f"Null counts: {null_counts}")

    print("\nInspecting building_meta.csv:")
    b_meta = pd.read_csv(os.path.join(DATA_DIR, "building_meta.csv"))
    print(b_meta.head(10))
    print("Building meta categories:", b_meta['category'].value_counts(dropna=False).to_dict())

    print("\nInspecting campus_meta.csv:")
    c_meta = pd.read_csv(os.path.join(DATA_DIR, "campus_meta.csv"))
    print(c_meta)

    print("\nInspecting calender.csv:")
    cal = pd.read_csv(os.path.join(DATA_DIR, "calender.csv"))
    print(cal.head())
    print("Calendar stats:\n", cal.describe())

    print("\nInspecting weather_data.csv sample:")
    w_head = pd.read_csv(os.path.join(DATA_DIR, "weather_data.csv"), nrows=20)
    print(w_head)

if __name__ == "__main__":
    deep_inspect()
