import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Graphura AI Social Intelligence", layout="wide")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Graphura Social Media Data.csv")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Ensure required numeric columns exist
    required_numeric = [
        "Likes", "Comments", "Shares", "Saves",
        "Views", "Reach", "Hashtag_Count"
    ]

    for col in required_numeric:
        if col not in df.columns:
            df[col] = 0

    return df


@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline.joblib")


data = load_data()
model = load_model()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
data["Engagement"] = (
    data["Likes"] +
    data["Comments"] +
    data["Shares"] +
    data["Saves"]
)

data["ER_by_Views"] = (data["Engagement"] / data["Views"].replace(0, 1)) * 100
data["ER_by_Reach"] = (data["Engagement"] / data["Reach"].replace(0, 1)) * 100
data["Day"] = data["Date"].dt.day_name()

st.title("📊 Graphura AI Social Intelligence Platform")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔵 LinkedIn Dashboard",
    "🟣 Instagram Dashboard",
    "🔮 AI Prediction",
    "📈 Forecasting",
    "⚔ Platform Comparison"
])


# ==========================================================
# DASHBOARD FUNCTION
# ==========================================================
def platform_dashboard(df, platform_name):

    st.subheader(f"{platform_name} Performance Dashboard")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Posts", df["Post ID"].nunique())
    col2.metric("Total Views", f"{df['Views'].sum():,}")
    col3.metric("Total Reach", f"{df['Reach'].sum():,}")
    col4.metric("Total Engagement", f"{df['Engagement'].sum():,}")
    col5.metric("Avg ER (%)", f"{df['ER_by_Views'].mean():.2f}")

    st.markdown("---")

    colA, colB = st.columns(2)

    with colA:
        pie_df = df.groupby("Post type")["Views"].sum().reset_index()

        fig = px.pie(
            pie_df,
            names="Post type",
            values="Views",
            hole=0.4,
            title="Views Distribution by Post Type"
        )

        st.plotly_chart(fig, use_container_width=True)

    with colB:
        bar_df = df.groupby("Post type")["Reach"].sum().reset_index()

        fig = px.bar(
            bar_df,
            x="Post type",
            y="Reach",
            text="Reach",
            color="Post type"
        )

        fig.update_traces(textposition="outside")
        fig.update_layout(title="Total Reach by Post Type")

        st.plotly_chart(fig, use_container_width=True)

    trend = df.groupby("Date")["Views"].sum().reset_index()

    fig = px.line(
        trend,
        x="Date",
        y="Views",
        markers=True,
        title="Views Trend Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="Views",
        y="ER_by_Views",
        size="Reach",
        color="Post type",
        hover_data=["Likes", "Comments", "Shares"],
        title="Engagement Rate vs Views"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top 10 Posts by Engagement")

    top_posts = (
        df.sort_values("Engagement", ascending=False)
        .head(10)[[
            "Post ID",
            "Post type",
            "Views",
            "Reach",
            "Engagement",
            "ER_by_Views"
        ]]
    )

    st.dataframe(top_posts, use_container_width=True)


# ==========================================================
# LINKEDIN TAB
# ==========================================================
with tab1:
    df_li = data[data["Platform"].str.lower() == "linkedin"]
    platform_dashboard(df_li, "LinkedIn")


# ==========================================================
# INSTAGRAM TAB
# ==========================================================
with tab2:
    df_ig = data[data["Platform"].str.lower() == "instagram"]
    platform_dashboard(df_ig, "Instagram")


# ==========================================================
# AI PREDICTION TAB
# ==========================================================
with tab3:

    st.subheader("🔮 Predict Post Performance")

    platform = st.selectbox("Platform", data["Platform"].unique())
    post_type = st.selectbox("Post Type", data["Post type"].unique())

    views = st.number_input("Views", 0, 1000000, 1000)
    reach = st.number_input("Reach", 0, 1000000, 900)
    likes = st.number_input("Likes", 0, 100000, 100)
    shares = st.number_input("Shares", 0, 100000, 10)
    follows = st.number_input("Follows", 0, 100000, 5)
    comments = st.number_input("Comments", 0, 100000, 10)
    saves = st.number_input("Saves", 0, 100000, 5)
    hashtag_count = st.number_input("Hashtag Count", 0, 50, 5)

    if st.button("Predict"):

        input_df = pd.DataFrame([{
            "Platform": platform,
            "Post type": post_type,
            "Views": views,
            "Reach": reach,
            "Likes": likes,
            "Shares": shares,
            "Follows": follows,
            "Comments": comments,
            "Saves": saves,
            "Hashtag_Count": hashtag_count
        }])

        # Align with model expected features safely
        try:
            required_cols = model.feature_names_in_
            input_df = input_df.reindex(columns=required_cols, fill_value=0)
        except:
            pass

        prediction = model.predict(input_df)[0]

        # Safe display (works for regression & classification)
        try:
            prediction_display = round(float(prediction), 2)
        except:
            prediction_display = prediction

        st.success(f"Predicted Performance: {prediction_display} 🚀")


# ==========================================================
# FORECASTING TAB
# ==========================================================
with tab4:

    st.subheader("📈 30-Day Views Forecasting")

    forecast_platform = st.selectbox(
        "Select Platform",
        data["Platform"].unique()
    )

    df_forecast = data[data["Platform"] == forecast_platform].dropna(subset=["Date"])
    df_forecast = df_forecast.sort_values("Date")

    daily = df_forecast.groupby(df_forecast["Date"].dt.date)["Views"].sum().reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Ordinal"] = daily["Date"].map(pd.Timestamp.toordinal)

    X = daily[["Ordinal"]]
    y = daily["Views"]

    lr = LinearRegression()
    lr.fit(X, y)

    future_dates = pd.date_range(daily["Date"].max(), periods=30)
    future_df = pd.DataFrame({"Date": future_dates})
    future_df["Ordinal"] = future_df["Date"].map(pd.Timestamp.toordinal)
    future_df["Predicted"] = lr.predict(future_df[["Ordinal"]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Views"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted"], mode="lines", name="Forecast"))

    fig.update_layout(title="30-Day Views Forecast")

    st.plotly_chart(fig, use_container_width=True)
    st.success("Forecast Generated Successfully 🚀")


# ==========================================================
# PLATFORM COMPARISON TAB
# ==========================================================
with tab5:

    st.subheader("⚔ Platform Comparison")

    compare_df = data.groupby("Platform").agg({
        "Views": "sum",
        "Reach": "sum",
        "Engagement": "sum",
        "ER_by_Views": "mean"
    }).reset_index()

    fig = px.bar(
        compare_df,
        x="Platform",
        y="Views",
        text="Views",
        color="Platform",
        title="Total Views by Platform"
    )

    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
