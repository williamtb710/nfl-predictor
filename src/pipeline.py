import nflreadpy as nfl
import pandas as pd
import os

SEASONS = list(range(2020, 2026))  # now includes 2025
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def fetch_weekly_stats():
    print("Fetching weekly player stats...")
    df = nfl.load_player_stats(SEASONS)
    df = df.to_pandas()  # nflreadpy uses Polars, convert to pandas
    df.to_parquet(f"{RAW_DIR}/weekly_stats.parquet", index=False)
    print(f"  Done — {len(df):,} rows saved")
    return df

def fetch_schedules():
    print("Fetching schedules...")
    df = nfl.load_schedules(SEASONS)
    df = df.to_pandas()
    df.to_parquet(f"{RAW_DIR}/schedules.parquet", index=False)
    print(f"  Done — {len(df):,} rows saved")
    return df

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    fetch_weekly_stats()
    fetch_schedules()
    print("\nAll data fetched successfully.")