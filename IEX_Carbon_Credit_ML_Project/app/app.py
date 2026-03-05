# ==========================================================
# IEX CARBON TRADING AI PLATFORM
# (Aligned with Training Pipeline)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="IEX Carbon Trading Intelligence",
    page_icon="⚡",
    layout="wide"
)

# ----------------------------------------------------------
# THEME
# ----------------------------------------------------------
st.markdown("""
<style>
.main-title{
    color:#2A447F;
    font-size:38px;
    font-weight:800;
}
.section{
    color:#2A447F;
    font-size:22px;
    font-weight:600;
}
.stButton>button{
    background:#2A447F;
    color:white;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline.joblib")

model = load_model()

# ==========================================================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ==========================================================
EPS = 1e-6

fuel_factor = {
    "Coal": 0.95,
    "Mixed Fuel": 0.75,
    "Natural Gas": 0.5,
    "Renewable": 0.25
}

def engineer_features(df):

    df["Emission_Gap"] = (
        df["Emission_Produced_Tco2"]
        - df["Emission_Allowance_Tco2"]
    )

    df["Compliance_Pressure"] = (
        df["Emission_Gap"]
        * df["Carbon_Price_Usd_Per_T"]
    )

    df["Cost_per_MWh"] = np.where(
        df["Energy_Demand_Mwh"] > 0,
        df["Compliance_Cost_Usd"] /
        (df["Energy_Demand_Mwh"] + EPS),
        0
    )

    df["Fuel_Carbon_Factor"] = (
        df["Fuel_Type"]
        .map(fuel_factor)
        .fillna(0.5)
    )

    return df


# ==========================================================
# HEADER
# ==========================================================
st.markdown('<div class="main-title">⚡ IEX Carbon Trading AI</div>', unsafe_allow_html=True)
st.caption("AI-powered Carbon Market Decision Intelligence")

tab1, tab2 = st.tabs([
    "🔮 AI Prediction",
    "📊 Batch Prediction"
])

# ==========================================================
# TAB 1 — SINGLE PREDICTION
# ==========================================================
with tab1:

    st.markdown('<div class="section">Predict Trading Decision</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        energy = st.number_input("Energy Demand (MWh)", 0.0, 100000.0, 1000.0)
        carbon_price = st.number_input("Carbon Price ($/t)", 0.0, 500.0, 40.0)
        credits = st.number_input("Credits Traded", 0.0, 100000.0, 100.0)

    with col2:
        emission_prod = st.number_input("Emission Produced", 0.0, 100000.0, 900.0)
        emission_allow = st.number_input("Emission Allowance", 0.0, 100000.0, 850.0)

    with col3:
        compliance_cost = st.number_input("Compliance Cost", 0.0, 1000000.0, 50000.0)
        fuel_type = st.selectbox("Fuel Type",
                                 ["Coal","Mixed Fuel","Natural Gas","Renewable"])
        scenario = st.selectbox("Optimization Scenario",
                                ["Aggressive","Balanced","Conservative"])

    if st.button("⚡ Predict Transaction"):

        input_df = pd.DataFrame([{
            "Energy_Demand_Mwh": energy,
            "Carbon_Price_Usd_Per_T": carbon_price,
            "Credits_Traded_Tco2": credits,
            "Emission_Produced_Tco2": emission_prod,
            "Emission_Allowance_Tco2": emission_allow,
            "Compliance_Cost_Usd": compliance_cost,
            "Fuel_Type": fuel_type,
            "Optimization_Scenario": scenario
        }])

        # --- FEATURE ENGINEERING ---
        input_df = engineer_features(input_df)

        # Keep only model features
        model_features = [
            "Energy_Demand_Mwh",
            "Carbon_Price_Usd_Per_T",
            "Credits_Traded_Tco2",
            "Emission_Gap",
            "Compliance_Pressure",
            "Cost_per_MWh",
            "Fuel_Carbon_Factor",
            "Optimization_Scenario"
        ]

        input_df = input_df[model_features]

        prediction = model.predict(input_df)[0]

        label_map = {0:"🟢 BUY",1:"🟡 HOLD",2:"🔴 SELL"}

        st.success(f"Predicted Decision: {label_map[prediction]}")

# ==========================================================
# TAB 2 — BATCH PREDICTION
# ==========================================================
with tab2:

    st.markdown('<div class="section">Upload CSV for Bulk Prediction</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload dataset")

    if file:

        df = pd.read_csv(file)

        df = engineer_features(df)

        model_features = [
            "Energy_Demand_Mwh",
            "Carbon_Price_Usd_Per_T",
            "Credits_Traded_Tco2",
            "Emission_Gap",
            "Compliance_Pressure",
            "Cost_per_MWh",
            "Fuel_Carbon_Factor",
            "Optimization_Scenario"
        ]

        df_model = df[model_features]

        df["Prediction"] = model.predict(df_model)

        st.dataframe(df.head())

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            "predictions.csv"
        )