# IEX Carbon Credit ML Project

## Overview
This project delivers an end-to-end carbon trading intelligence workflow: SQL ingestion, exploratory analysis, feature engineering, machine learning classification, dashboarding, and Streamlit-based inference.

## Business Problem
Predict optimal carbon-market trade action (`BUY`, `HOLD`, `SELL`) from emissions, compliance pressure, pricing, and fuel/operations context.

## What Is In This Repository
- `app.py`: current Streamlit app source at root.
- `app/app.py`: structured app copy used for deployment organization.
- `data/raw/IEX_Carbon_Trading_Dataset.csv`: canonical dataset copy.
- `sql/Carbon_Trading_DB.sql`: MySQL setup + robust CSV import script.
- `notebooks/`: EDA, business queries, feature engineering, training, and evaluation notebooks.
- `models/best_model_pipeline.joblib`: trained model artifact used by app.
- `dashboards/IEX_Dashboard.pbix`: business dashboard file.

## Key File Content Summary
- Dataset columns include company profile, energy demand, fuel type, emissions produced/allowed, carbon price, credits traded, compliance cost, optimization scenario, savings, and transaction label.
- App includes single prediction and batch CSV prediction flows with aligned feature engineering (`Emission_Gap`, `Compliance_Pressure`, `Cost_per_MWh`, `Fuel_Carbon_Factor`).
- Notebooks include 100-cell business EDA and dedicated research/training notebooks for classification models.

## Folder Map
- `data/`: raw + processed data layers.
- `notebooks/`: standardized (`01-07`) and source-named notebooks.
- `src/`: modular Python utilities for preprocessing, features, training, evaluation, prediction.
- `models/`: serialized ML artifacts.
- `app/`: Streamlit app folder.
- `sql/`: database scripts.
- `dashboards/`: Power BI assets.
- `reports/`: report outputs and figure assets.
