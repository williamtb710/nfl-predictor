import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

os.chdir("c:/Users/willi/nfl-predictor")

#load model and data
@st.cache_resource
def load_model():
    with open("models/xgboost_wr.pkl", "rb") as f:
        return pickle.load(f)
    
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/wr_features.parquet")

model = load_model()
df = load_data()

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

st.title("NFL Wide Receiver Performance Predictor")
st.markdown("Predict receiving yards for any WR in the dataset using XGBoost + SHAP explainability.")

# Sidebar controls
st.sidebar.header("Select a game to analyze")

players = sorted(df['player_name'].unique())
selected_player = st.sidebar.selectbox("Player", players)

player_df = df[df['player_name'] == selected_player].sort_values(['season', 'week'])
seasons = sorted(player_df['season'].unique(), reverse=True)
selected_season = st.sidebar.selectbox("Season", seasons)

season_df = player_df[player_df['season'] == selected_season]
weeks = sorted(season_df['week'].unique())
selected_week = st.sidebar.selectbox("Week", weeks)

# Get the selected row
row = season_df[season_df['week'] == selected_week]

if len(row) == 0:
    st.warning("No data found for this selection.")
else:
    row = row.iloc[0]
    X = pd.DataFrame([row[FEATURE_COLS]])

    predicted = model.predict(X)[0]
    actual = row['receiving_yards']
    opponent = row['opponent_team']
    is_home = "Home" if row['is_home'] == 1 else "Away"

    # Main metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted yards", f"{predicted:.0f}")
    col2.metric("Actual yards", f"{actual:.0f}")
    col3.metric("Opponent", f"{opponent} ({is_home})")

    st.divider()

    # SHAP waterfall
    st.subheader("Why did the model predict this?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # Player history table
    st.subheader(f"{selected_player} — {selected_season} season")
    history = season_df[['week', 'opponent_team', 'receiving_yards_roll3', 'receiving_yards_roll5', 'receiving_yards']].copy()
    history.columns = ['Week', 'Opponent', 'Yards (roll3)', 'Yards (roll5)', 'Actual Yards']
    st.dataframe(history.set_index('Week'), use_container_width=True)