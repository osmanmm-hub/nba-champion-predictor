# NBA Championship Predictor

A machine learning app that predicts which NBA team will win the 2025-26 Championship using live stats.

## What it does
- Pulls **live 2025-26 NBA team stats** from the NBA Stats API
- Uses models trained on **25 years of historical data** (2000–2025)
- Predicts **championship probability** for all 30 teams
- Explains predictions with **SHAP visualizations**

## How to use
Visit the app and explore 4 tabs:
- 📋 **Executive Summary** — overview of the methodology and key findings
- 📊 **Historical Analysis** — what separates champions from the rest
- 🤖 **Model Performance** — comparison of 4 ML models
- 🏆 **Live Predictions** — current 2025-26 championship probabilities + SHAP explanation for any team

## Models Used
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network (Keras)

## Tech Stack
Python · Streamlit · XGBoost · SHAP · NBA API · scikit-learn · pandas

## Data Source
Live data via [nba_api](https://github.com/swar/nba_api) — no API key required
