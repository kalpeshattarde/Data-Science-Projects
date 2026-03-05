import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline.joblib")

model = load_model()

# -----------------------------
# App UI
# -----------------------------
st.title("📊 Social Media Post Performance Predictor")
st.write("Predict whether a post will have **Low**, **Medium**, or **High** performance.")

st.sidebar.header("📌 Post Details")

# -----------------------------
# User Inputs (SAFE)
# -----------------------------
duration = st.sidebar.number_input("Duration (sec)", min_value=0, value=30)
views = st.sidebar.number_input("Views", min_value=0, value=1000)
reach = st.sidebar.number_input("Reach", min_value=0, value=900)
likes = st.sidebar.number_input("Likes", min_value=0, value=100)
shares = st.sidebar.number_input("Shares", min_value=0, value=10)
follows = st.sidebar.number_input("Follows", min_value=0, value=5)
comments = st.sidebar.number_input("Comments", min_value=0, value=10)
saves = st.sidebar.number_input("Saves", min_value=0, value=5)

hashtag_count = st.sidebar.number_input("Hashtag Count", min_value=0, value=3)

er_views = st.sidebar.number_input(
    "Engagement Rate by Views (%)", min_value=0.0, value=10.0
)
er_reach = st.sidebar.number_input(
    "Engagement Rate by Reach (%)", min_value=0.0, value=12.0
)

post_type = st.sidebar.selectbox(
    "Post Type", ["image", "carousel", "reel"]
)

# Boolean encoding (same as training)
posttype_image = post_type == "image"
posttype_carousel = post_type == "carousel"
posttype_reel = post_type == "reel"

# -----------------------------
# Build input DataFrame
# -----------------------------
input_data = pd.DataFrame([{
    "Duration (sec)": duration,
    "Views": views,
    "Reach": reach,
    "Likes": likes,
    "Shares": shares,
    "Follows": follows,
    "Comments": comments,
    "Saves": saves,
    "Hashtag_Count": hashtag_count,
    "Engagement Rate by Views (%)": er_views,
    "Engagement Rate by Reach (%)": er_reach,
    "Post type": post_type,
    "PostType_carousel": posttype_carousel,
    "PostType_image": posttype_image,
    "PostType_reel": posttype_reel
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Performance"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("📈 Prediction Result")
    st.success(f"**Predicted Performance:** {prediction}")

    st.subheader("🔍 Confidence")
    prob_df = pd.DataFrame({
        "Class": model.classes_,
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Class"))



streamlit run app.py