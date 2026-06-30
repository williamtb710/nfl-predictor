# NFL Wide Receiver Performance Predictor

A machine learning system that predicts NFL wide receiver receiving yards using XGBoost, trained on 2020–2025 season data with SHAP explainability.

**Live demo:** [link coming soon]

---

## Overview

Predicting individual player performance in the NFL is a genuinely hard problem — outcomes depend on player skill, team context, opponent strength, and in-game variance. This project builds an end-to-end ML pipeline that ingests raw NFL data, engineers meaningful features, trains and tunes multiple models, and serves predictions through an interactive web app.

---

## Results

| Model | MAE (yards) | RMSE (yards) |
|---|---|---|
| Baseline (3-game rolling avg) | 24.5 | 32.5 |
| Ridge Regression | 22.4 | 29.6 |
| XGBoost (default) | 22.7 | 29.9 |
| XGBoost (Optuna tuned) | 22.5 | 29.6 |

All models evaluated on held-out 2025 season data. Both ML models beat the naive baseline by ~2 yards MAE.

---

## Features

**Player history (rolling)**
- Receiving yards, targets, receptions — 3 and 5 game rolling averages
- Target share and WOPR rolling averages — measure of a player's share of the passing game
- Receiving EPA rolling averages — efficiency metric beyond raw volume
- Season-to-date averages for yards and targets

**Opponent strength**
- Average receiving yards allowed per game by each defense

**Game context**
- Home vs away flag

---

## Tech Stack

- **Data:** nflreadpy (nflverse), pandas, parquet
- **Modeling:** scikit-learn, XGBoost, Optuna (hyperparameter tuning)
- **Explainability:** SHAP
- **Experiment tracking:** MLflow
- **App:** Streamlit

---

## Project Structure

```
nfl-predictor/
├── data/
│   ├── raw/          # raw parquet files (not committed)
│   └── processed/    # feature matrix
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory data analysis
│   └── 02_shap.ipynb         # SHAP feature importance
├── src/
│   ├── pipeline.py   # data ingestion
│   ├── features.py   # feature engineering
│   ├── train.py      # model training + Optuna tuning
│   └── predict.py    # inference
├── app/
│   └── streamlit_app.py
└── models/
    └── xgboost_wr.pkl
```

---

## How to Run

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/williamtb710/nfl-predictor.git
cd nfl-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Fetch data**
```bash
python src/pipeline.py
```

**3. Build features**
```bash
python src/features.py
```

**4. Train models**
```bash
python src/train.py
```

**5. Launch the app**
```bash
streamlit run app/streamlit_app.py
```

---

## Key Findings

- **Season-long averages outperform recent form** as predictors — a player's established role matters more than a hot or cold streak
- **Target share is more stable than raw yards** — it measures usage independent of game script
- **Opponent defensive strength** is the third most important feature, validating that matchup matters
- **Home/away has minimal impact** on individual WR performance at the game level
- The model struggles with outlier performances (e.g. Ja'Marr Chase's 264-yard game predicted at 69 yards) — a known limitation of any regression model on high-variance outcomes

---

## What I'd Add Next

- Expand to RB, QB, and TE positions using the same pipeline
- Add Vegas implied team totals as a feature (strong proxy for passing volume)
- Add weather data for outdoor games
- CB matchup quality feature — map each WR to their likely coverage defender
- Retrain weekly during the season as new data becomes available

---

## Author

Built by William — Mathematics of Computation, UCLA