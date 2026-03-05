import streamlit as st
import pandas as pd
import joblib

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Smart Silage Pricing AI",
    layout="wide"
)

st.title("🌾 Smart Silage Pricing Intelligence System")

st.markdown("""
This AI model predicts **Price per Metric Ton of Silage** based on crop type,
season, packaging, customer segment, and order quantity.
""")

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline.joblib")

model = load_model()

# =========================================
# INPUT SECTION
# =========================================
st.subheader("Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    customer_type = st.selectbox(
        "Customer Type",
        ["Distributor","Dairy Farm","Co-operative","Individual Farmer"]
    )

    crop_type = st.selectbox(
        "Crop Type",
        ["Bajra","Jowar","Maize","Hybrid Mix"]
    )

with col2:

    season = st.selectbox(
        "Harvest Season",
        ["Kharif","Rabi","Summer"]
    )

    bagging = st.selectbox(
        "Bagging Type",
        ["Bags","Bale-25kg","Bale-50kg","Bulk"]
    )

with col3:

    moisture = st.slider(
        "Moisture Content %",
        50,75,63
    )

    quantity = st.number_input(
        "Quantity (MT)",
        1,200,30
    )

# =========================================
# FEATURE ENGINEERING (same logic as training)
# =========================================

if 62 <= moisture <= 68:
    silage_quality = "High Quality Silage"
else:
    silage_quality = "Average Quality Silage"

if quantity < 5:
    discount_rate = 0.025
elif quantity < 10:
    discount_rate = 0.05
elif quantity < 25:
    discount_rate = 0.10
else:
    discount_rate = 0.15

# =========================================
# PREDICTION
# =========================================

if st.button("Predict Price"):

    input_data = pd.DataFrame([{
        "Discount_rate": discount_rate,
        "Customer_Type": customer_type,
        "Crop_Type": crop_type,
        "Harvest_Season": season,
        "Bagging_Type": bagging,
        "Silage_Quality": silage_quality
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Price per MT: ₹ {round(prediction,2)}")

    total_value = prediction * quantity

    st.metric(
        "Estimated Order Value",
        f"₹ {round(total_value,2)}"
    )

# =========================================
# INFORMATION PANEL
# =========================================

st.markdown("---")

st.subheader("Model Insights")

st.markdown("""
This model uses **Gradient Boosting Regression** trained on historical silage trading data.

Key drivers of price:

• Crop Type  
• Harvest Season  
• Packaging Type  
• Customer Segment  
• Silage Quality  
• Quantity Discount  

The model helps businesses optimize **pricing strategy and supply chain decisions**.
""")