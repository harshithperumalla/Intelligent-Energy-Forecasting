import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def check_meters_and_weather():
    print("Reading building_consumption meter counts...")
    df = pd.read_csv(os.path.join(DATA_DIR, "building_consumption.csv"), usecols=['campus_id', 'meter_id', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    summary = df.groupby(['campus_id', 'meter_id']).agg(
        count=('timestamp', 'count'),
        min_date=('timestamp', 'min'),
        max_date=('timestamp', 'max')
    ).reset_index()
    
    print("Top 15 meters by reading count:")
    print(summary.sort_values(by='count', ascending=False).head(15))
    
    print("\nBottom 5 meters by reading count:")
    print(summary.sort_values(by='count', ascending=True).head(5))

    print("\nCampus distribution:")
    print(summary.groupby('campus_id')['meter_id'].count())

    print("\nChecking weather campuses...")
    # Sample weather campuses
    w_campuses = set()
    for chunk in pd.read_csv(os.path.join(DATA_DIR, "weather_data.csv"), usecols=['campus_id'], chunksize=1000000):
        w_campuses.update(chunk['campus_id'].unique())
    print("Campuses in weather_data:", sorted(list(w_campuses)))

if __name__ == "__main__":
    check_meters_and_weather()
