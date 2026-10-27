import os
import time
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def test_prep():
    t0 = time.time()
    print("Testing hourly weather aggregation...")
    # Read weather data with needed columns
    w_cols = ['campus_id', 'timestamp', 'air_temperature', 'apparent_temperature', 'relative_humidity', 'wind_speed']
    # Let's read weather in chunks and aggregate to hourly
    hourly_weather_list = []
    print("Reading weather data in chunks...")
    for chunk in pd.read_csv(os.path.join(DATA_DIR, "weather_data.csv"), usecols=w_cols, chunksize=1000000):
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        # group by campus_id and dt.floor('h')
        chunk['timestamp'] = chunk['timestamp'].dt.floor('h')
        agg = chunk.groupby(['campus_id', 'timestamp']).mean().reset_index()
        hourly_weather_list.append(agg)
        print(f"Processed chunk, accumulated {len(hourly_weather_list)} chunks...")

    df_weather = pd.concat(hourly_weather_list, ignore_index=True)
    df_weather_hourly = df_weather.groupby(['campus_id', 'timestamp']).mean().reset_index()
    print(f"Hourly weather shape: {df_weather_hourly.shape}, took {time.time() - t0:.2f}s")
    print(df_weather_hourly.head())

if __name__ == "__main__":
    test_prep()
