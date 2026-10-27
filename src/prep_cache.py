import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def prepare_dashboard_cache():
    print("Preparing dashboard cache for instantaneous Streamlit loading...")
    src_path = os.path.join(DATA_DIR, "featured_hourly_consumption.parquet")
    df = pd.read_parquet(src_path)

    # Filter to test set (2022) + recent 3 months of 2021 for continuous lag context
    active_df = df[df['timestamp'] >= '2021-10-01'].copy()
    active_meters = set(active_df['meter_id'].unique())
    all_meters = set(df['meter_id'].unique())
    missing_meters = all_meters - active_meters

    # For any decommissioned meters not in the recent window, append their last 500 records
    extra_dfs = []
    for m in missing_meters:
        m_slice = df[df['meter_id'] == m].sort_values('timestamp').tail(500)
        extra_dfs.append(m_slice)

    if extra_dfs:
        dash_df = pd.concat([active_df] + extra_dfs, ignore_index=True)
    else:
        dash_df = active_df

    dash_df = dash_df.sort_values(['campus_id', 'meter_id', 'timestamp']).reset_index(drop=True)
    dst_path = os.path.join(OUTPUT_DIR, "dashboard_data.parquet")
    dash_df.to_parquet(dst_path, index=False)
    print(f"Saved dashboard cache: {dst_path} ({os.path.getsize(dst_path)/(1024*1024):.2f} MB, {len(dash_df):,} rows across {dash_df['meter_id'].nunique()} buildings)")

if __name__ == "__main__":
    prepare_dashboard_cache()
