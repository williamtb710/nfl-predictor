import pandas as pd
import numpy as np
import os

os.chdir("c:/Users/willi/nfl-predictor")

def load_data():
    weekly = pd.read_parquet("data/raw/weekly_stats.parquet")
    schedules = pd.read_parquet("data/raw/schedules.parquet")
    return weekly, schedules

def build_wr_features(weekly, schedules):
    #filter for WRs + regular season only
    wr = weekly[(weekly['position'] == 'WR') & (weekly['season_type'] == 'REG')].copy()

    #drop rows with missing advanced metrics
    wr = wr.dropna(subset=['target_share', 'air_yards_share', 'receiving_epa', 'wopr'])

    #sort so rolling calculations trend in the right direction
    wr = wr.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    #rolling player features
    grp = wr.groupby('player_id')

    #rolling averages for last 3 and 5 games
    for col in ['receiving_yards', 'targets', 'receptions', 'target_share', 'wopr', 'receiving_epa']:
        wr[f'{col}_roll3'] = grp[col].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        wr[f'{col}_roll5'] = grp[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    #season to date averages
    wr['yards_season_avg'] = grp['receiving_yards'].transform(lambda x: x.shift(1).expanding().mean())
    wr['targets_season_avg'] = grp['targets'].transform(lambda x: x.shift(1).expanding().mean())

    #opponent defensive strength
    #calculate how many yards each team allows to wide receivers on a per-game basis
    opp_strength = wr.groupby(['season', 'opponent_team'])['receiving_yards'].mean().reset_index()
    opp_strength.columns = ['season', 'opponent_team', 'opp_yards_allowed_per_game']

    wr = wr.merge(opp_strength, on=['season', 'opponent_team'], how='left')

    # --- Home/away from schedules ---
    # Build a lookup: for each game, which team is home
    schedule_lookup = schedules[['season', 'week', 'home_team', 'away_team']].copy()

    def get_home_flag(row):
        match = schedule_lookup[
            (schedule_lookup['season'] == row['season']) &
            (schedule_lookup['week'] == row['week']) &
            ((schedule_lookup['home_team'] == row['recent_team']) |
             (schedule_lookup['away_team'] == row['recent_team']))
        ]
        if len(match) == 0:
            return np.nan
        return 1 if match.iloc[0]['home_team'] == row['recent_team'] else 0

    wr['is_home'] = wr.apply(get_home_flag, axis=1)

    # --- Select final feature columns ---
    feature_cols = [
        'player_id', 'player_name', 'season', 'week',
        'recent_team', 'opponent_team', 'is_home',
        'receiving_yards_roll3', 'receiving_yards_roll5',
        'targets_roll3', 'targets_roll5',
        'receptions_roll3', 'receptions_roll5',
        'target_share_roll3', 'target_share_roll5',
        'wopr_roll3', 'wopr_roll5',
        'receiving_epa_roll3', 'receiving_epa_roll5',
        'yards_season_avg', 'targets_season_avg',
        'opp_yards_allowed_per_game',
        'receiving_yards'  # target variable
    ]

    wr = wr[feature_cols].dropna()

    return wr

if __name__ == "__main__":
    weekly, schedules = load_data()
    features = build_wr_features(weekly, schedules)
    print(f"Feature matrix shape: {features.shape}")
    print(f"\nFirst few rows:")
    print(features.head())
    features.to_parquet("data/processed/wr_features.parquet", index=False)
    print("\nSaved to data/processed/wr_features.parquet")