#app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Runs Predictor",
    page_icon="🏏",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("dataset/cricket_data.csv", parse_dates=["match_date"])

bundle = load_model()
df     = load_data()
model    = bundle["model"]
encoders = bundle["encoders"]

# ── Sidebar — Prediction Form ─────────────────────────────────
st.sidebar.title("🏏 Match Details")

players   = list(encoders["player_name"].classes_)
opponents = list(encoders["opponent_team"].classes_)
venues    = list(encoders["venue"].classes_)

player     = st.sidebar.selectbox("Select Batsman",      players)
opponent   = st.sidebar.selectbox("Select Opponent Team", opponents)
venue      = st.sidebar.selectbox("Select Venue",         venues)
recent_avg = st.sidebar.slider("Recent Avg Runs (Last 5)", 0.0, 100.0, 35.0, step=0.5)

if st.sidebar.button("🎯 Predict Runs", use_container_width=True):
    p_enc = encoders["player_name"].transform([player])[0]
    o_enc = encoders["opponent_team"].transform([opponent])[0]
    v_enc = encoders["venue"].transform([venue])[0]
    X     = np.array([[p_enc, o_enc, v_enc, recent_avg]])
    pred  = max(0, round(float(model.predict(X)[0])))
    st.sidebar.success(f"Predicted Runs: **{pred}**")
    st.sidebar.caption(f"{player} vs {opponent}")

# ── Main Page ─────────────────────────────────────────────────
st.title("🏏 Cricket Player Runs Prediction System")
st.caption("IPL Batting Analytics | Linear Regression")

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records",   f"{len(df):,}")
c2.metric("Players",         df["player_name"].nunique())
c3.metric("Venues",          df["venue"].nunique())
c4.metric("Avg Runs/Innings",f"{df['runs_scored'].mean():.1f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🎯 Model Results", "📋 Data"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Run Scorers")
        top = df.groupby("player_name")["runs_scored"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top)

    with col2:
        st.subheader("Venue-wise Avg Runs")
        venue_avg = df.groupby("venue")["runs_scored"].mean().sort_values(ascending=False)
        venue_avg.index = venue_avg.index.str.split(",").str[0]
        st.bar_chart(venue_avg)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Runs Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["runs_scored"], bins=35, color="#4C9BE8", edgecolor="white")
        ax.set_xlabel("Runs"); ax.set_ylabel("Frequency")
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Opponent-wise Avg Runs")
        opp_avg = df.groupby("opponent_team")["runs_scored"].mean().sort_values(ascending=False)
        st.bar_chart(opp_avg)

with tab2:
    st.subheader("Actual vs Predicted Runs")
    try:
        preds_df = pd.read_csv("outputs/predictions.csv")
        fig, ax  = plt.subplots(figsize=(6, 5))
        ax.scatter(preds_df["actual"], preds_df["predicted"], alpha=0.4, s=15, color="#4C9BE8")
        lim = preds_df.max().max() + 5
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect Prediction")
        ax.set_xlabel("Actual Runs"); ax.set_ylabel("Predicted Runs")
        ax.legend()
        col1, col2 = st.columns([1, 1])
        col1.pyplot(fig); plt.close()

        from sklearn.metrics import mean_absolute_error, r2_score
        import math
        mae  = mean_absolute_error(preds_df["actual"], preds_df["predicted"])
        rmse = math.sqrt(((preds_df["actual"] - preds_df["predicted"])**2).mean())
        r2   = r2_score(preds_df["actual"], preds_df["predicted"])
        with col2:
            st.metric("MAE",  f"{mae:.2f} runs")
            st.metric("RMSE", f"{rmse:.2f} runs")
            st.metric("R² Score", f"{r2:.4f}")
    except FileNotFoundError:
        st.warning("Run `python main.py` first to generate predictions.")

with tab3:
    st.subheader("Raw Dataset")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Showing first 100 of {len(df):,} records")