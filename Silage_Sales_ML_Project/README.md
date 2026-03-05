# Silage Sales ML Project

## Overview
End-to-end business analytics and ML project for silage pricing and order-value intelligence.

## Current Structured State
This project is already folderized. Existing files are retained and synchronized in-place.

## Key Assets
- `app/app.py`: Streamlit pricing app (interactive input + prediction).
- `data/raw/Pancham_Silage_Factory.csv`: source transaction dataset.
- `sql/Pancham_Silage_Factory_DB.sql`: schema + bulk CSV import + cleanup rules.
- `notebooks/01-07`: EDA, business SQL analysis, feature engineering, training, and reporting notebooks.
- `models/best_model_pipeline.joblib`: serialized model artifact.
- `dashboards/Pancham_Dashboard.pbix`: BI dashboard.

## Business Objective
Predict silage price per MT and support sales planning using customer segment, crop type, seasonality, moisture/quality, bagging type, and volume-based discount logic.
