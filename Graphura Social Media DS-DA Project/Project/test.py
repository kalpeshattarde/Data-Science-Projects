import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Check current directory (debug)
# -----------------------------
st.write("📂 App working directory:", os.getcwd())

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best_model_v1.joblib")
    return joblib.load(model_path)

model = load_model()

st.success("✅ Model loaded successfully")


st.header("🔮 Quick Test Prediction")

test_input = pd.DataFrame([{
    "Duration (sec)": 30,
    "Views": 1000,
    "Reach": 900,
    "Likes": 120,
    "Shares": 10,
    "Follows": 5,
    "Comments": 15,
    "Saves": 20,
    "Hashtag_Count": 4,
    "Engagement Rate by Views (%)": 12.0,
    "Engagement Rate by Reach (%)": 15.0,
    "Post type": "image",
    "PostType_carousel": False,
    "PostType_image": True,
    "PostType_reel": False
}])

if st.button("Run Test"):
    pred = model.predict(test_input)
    st.write("Prediction:", pred)
