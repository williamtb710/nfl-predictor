import nfl_data_py as nfl
import pandas as pd
import os

SEASONS = list(range(2020, 2025))
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def fetch_weekly_stats():
    print("Fetching weekly player stats...")
    df = nfl.import_weekly_data(SEASONS)
    df.to_parquet(f"{RAW_DIR}/weekly_stats.parquet", index=False)
    print(f" Done - {len(df):,} rows saved")
    return df

def fetch_schedules():
    print("Fetching schedules...")
    df = nfl.import_schedules(SEASONS)
    df.to_parquet(f"{RAW_DIR}/schedules.parquet", index=False)
    print(f" Done - {len(df):,} rows saved")
    return df

def fetch_rosters():
    print("Fetching rosters...")
    df = nfl.import_weekly_rosters(SEASONS)
    df.to_parquet(f"{RAW_DIR}/rosters.parquet", index=False)
    print(f" Done - {len(df):,} rows saved")
    return df

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    fetch_weekly_stats()
    fetch_schedules()
    fetch_rosters()
    print("\nAll data fetched successfully.")
