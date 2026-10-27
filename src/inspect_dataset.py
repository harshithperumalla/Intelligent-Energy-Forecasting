import os
import glob
import pandas as pd
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect_all():
    print(f"Inspecting files in {DATA_DIR}...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    summary = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"\n==========================================")
        print(f"FILE: {fname} ({fsize_mb:.2f} MB)")
        
        # Read header and first 5 rows
        df_head = pd.read_csv(fpath, nrows=5)
        cols = list(df_head.columns)
        print(f"Columns ({len(cols)}): {cols}")
        print("Sample data:")
        print(df_head.to_dict(orient="records")[:2])
        
        # Check total rows efficiently
        line_count = 0
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                line_count += 1
        total_rows = line_count - 1 if line_count > 0 else 0
        print(f"Total Rows: {total_rows}")

        # Check datetime columns or specific columns
        date_col = None
        for c in cols:
            if any(k in c.lower() for k in ["date", "time", "timestamp"]):
                date_col = c
                break
        
        date_range = None
        if date_col:
            try:
                # Sample head and tail to determine date range
                first_row = pd.read_csv(fpath, nrows=1)
                # To read last few rows efficiently
                tail_df = pd.read_csv(fpath, skiprows=range(1, max(1, total_rows - 5)), nrows=5)
                first_date = str(first_row[date_col].values[0])
                last_date = str(tail_df[date_col].values[-1])
                date_range = f"{first_date} to {last_date}"
                print(f"Identified Date Column: '{date_col}', Range: {date_range}")
            except Exception as e:
                print(f"Date check error: {e}")

        summary[fname] = {
            "size_mb": round(fsize_mb, 2),
            "rows": total_rows,
            "columns": cols,
            "sample_dtypes": {k: str(v) for k, v in df_head.dtypes.items()},
            "date_column": date_col,
            "date_range": date_range
        }

    with open(os.path.join(OUTPUT_DIR, "data_audit_summary.json"), "w") as out_f:
        json.dump(summary, out_f, indent=2)

    print("\nData audit summary saved to outputs/data_audit_summary.json")

if __name__ == "__main__":
    inspect_all()
