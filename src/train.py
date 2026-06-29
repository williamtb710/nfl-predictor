import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

os.chdir("c:/Users/willi/nfl-predictor")

FEATURE_COLS = [
    'receiving_yards_roll3', 'receiving_yards_roll5',
    'targets_roll3', 'targets_roll5',
    'receptions_roll3', 'receptions_roll5',
    'target_share_roll3', 'target_share_roll5',
    'wopr_roll3', 'wopr_roll5',
    'receiving_epa_roll3', 'receiving_epa_roll5',
    'yards_season_avg', 'targets_season_avg',
    'opp_yards_allowed_per_game', 'is_home'
]

TARGET = 'receiving_yards'

def load_features():
    df = pd.read_parquet("data/processed/wr_features.parquet")
    print(f"Loaded {len(df):,} rows")
    return df

def train_test_split_by_season(df, test_season=2024):
    """Always split by season — never random split on time series data."""
    train = df[df['season'] < test_season].copy()
    test = df[df['season'] == test_season].copy()
    print(f"Train: {len(train):,} rows ({df['season'].min()}–{test_season-1})")
    print(f"Test:  {len(test):,} rows ({test_season})")
    return train, test

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:20s} — MAE: {mae:.1f} yards  |  RMSE: {rmse:.1f} yards")
    return mae, rmse

def train_baseline(train, test):
    """Baseline: predict next game = 3-game rolling average. No ML needed."""
    y_pred = test['receiving_yards_roll3']
    y_true = test[TARGET]
    return evaluate("Baseline (roll3)", y_true, y_pred)

def train_ridge(train, test):
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae, rmse = evaluate("Ridge regression", y_test, y_pred)

    with mlflow.start_run(run_name="ridge"):
        mlflow.log_param("alpha", 1.0)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")

    return model, mae, rmse

def train_xgboost(train, test):
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]

    params = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    mae, rmse = evaluate("XGBoost", y_test, y_pred)

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(model, "model")

    os.makedirs("models", exist_ok=True)
    with open("models/xgboost_wr.pkl", "wb") as f:
        pickle.dump(model, f)
    print("  Model saved to models/xgboost_wr.pkl")

    return model, mae, rmse

if __name__ == "__main__":
    mlflow.set_experiment("nfl-wr-predictor")

    df = load_features()
    train, test = train_test_split_by_season(df, test_season=2025)

    print("\n--- Model Comparison ---")
    train_baseline(train, test)
    ridge_model, _, _ = train_ridge(train, test)
    xgb_model, _, _ = train_xgboost(train, test)

    print("\nDone. Check MLflow UI with: mlflow ui")