"""
preprocessing.py
----------------
DATA PREPROCESSING FOR CRICKET RUNS PREDICTION

🏏 Cricket Analogy:
Before an analyst can predict Rohit Sharma's next score, he reviews
the scorebook — cleans smudged entries, converts "DNB" to 0, and
groups innings by conditions. That's exactly what we do here.

Steps:
  1. Load the raw CSV (our scorebook)
  2. Fix / drop missing values
  3. Engineer "Recent Average" — the batter's form over last 5 innings
  4. Label-encode categorical columns (player, opponent, venue)
  5. Return a clean feature matrix X and target vector y
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ──────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str = "dataset/cricket_data.csv") -> pd.DataFrame:
    """
    Load the CSV scorecard.
    Like opening the match register at the start of a tournament.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ Dataset not found at '{filepath}'.\n"
            "   Run generate_dataset.py first, or place your CSV there."
        )
    df = pd.read_csv(filepath, parse_dates=["match_date"])
    print(f"✅ Loaded {len(df):,} records from '{filepath}'")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values.
    🏏 Analogy: If a player's score is missing (rain stoppage?),
    we fill it with the median so it doesn't skew the analysis.
    """
    before = len(df)
    # Drop rows where the target (runs_scored) is unknown
    df = df.dropna(subset=["runs_scored"])

    # Fill numeric NaNs with median
    num_cols = ["balls_faced", "strike_rate", "fours", "sixes"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaNs with mode
    cat_cols = ["player_name", "opponent_team", "venue"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    after = len(df)
    print(f"✅ Missing-value handling: {before - after} rows dropped → {after:,} remain")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
def create_recent_avg(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Feature Engineering: Recent Average Runs (last N innings).

    🏏 Analogy: A selector always checks a player's 'form' —
    the average over the last 5 games. If Shubman Gill scored
    [80, 60, 45, 70, 55] in his last 5 games, his recent average
    is 62. That's a far better predictor than his all-time average.

    We compute this per player, sorted by date, using a
    rolling mean shifted by 1 (so we never leak future scores).
    """
    df = df.sort_values(["player_name", "match_date"]).reset_index(drop=True)

    df["recent_avg_runs"] = (
        df.groupby("player_name")["runs_scored"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # For a player's very first record there's no history → use their global mean
    global_mean = df["runs_scored"].mean()
    df["recent_avg_runs"] = df["recent_avg_runs"].fillna(global_mean).round(2)

    print(f"✅ 'recent_avg_runs' feature created (rolling {window}-match window)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame, save_encoders: bool = True):
    """
    Label-encode player_name, opponent_team, venue.

    🏏 Analogy: Our model only understands numbers, not names.
    So we assign each player an ID — Virat Kohli → 5, Rohit Sharma → 12.
    We save those mappings (encoders) so we can translate back later.
    """
    encoders = {}
    for col in ["player_name", "opponent_team", "venue"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"   ↳ '{col}' encoded → {len(le.classes_)} unique values")

    if save_encoders:
        os.makedirs(".", exist_ok=True)
        with open("encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        print("✅ Label encoders saved → encoders.pkl")

    return df, encoders


# ──────────────────────────────────────────────────────────────────────────────
def get_features_target(df: pd.DataFrame):
    """
    Split the dataframe into features (X) and target (y).

    Features:
      - player_name_enc    → who is batting
      - opponent_team_enc  → who is bowling
      - venue_enc          → where is the match
      - recent_avg_runs    → player's current form

    Target:
      - runs_scored  → what we want to predict
    """
    feature_cols = [
        "player_name_enc",
        "opponent_team_enc",
        "venue_enc",
        "recent_avg_runs",
    ]
    X = df[feature_cols].values
    y = df["runs_scored"].values
    print(f"✅ Features shape: {X.shape}  |  Target shape: {y.shape}")
    return X, y, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
def preprocess_pipeline(filepath: str = "dataset/cricket_data.csv"):
    """
    Master pipeline: load → clean → engineer → encode → split.
    Call this from train_model.py for a one-shot preprocessing run.
    """
    print("\n" + "="*60)
    print("  🏏  CRICKET RUNS PREDICTION — DATA PREPROCESSING")
    print("="*60 + "\n")

    df = load_data(filepath)

    print("\n📋 Dataset Overview:")
    print(f"   Shape      : {df.shape}")
    print(f"   Columns    : {list(df.columns)}")
    print(f"   Date Range : {df['match_date'].min().date()} → {df['match_date'].max().date()}")
    print(f"\n{df.head(3).to_string()}\n")

    print("\n🔧 Step 1 — Handling Missing Values")
    df = handle_missing(df)

    print("\n🔧 Step 2 — Engineering 'Recent Average Runs'")
    df = create_recent_avg(df)

    print("\n🔧 Step 3 — Encoding Categorical Columns")
    df, encoders = encode_categoricals(df)

    print("\n🔧 Step 4 — Extracting Features & Target")
    X, y, feature_cols = get_features_target(df)

    print("\n✅ Preprocessing complete!\n")
    return df, X, y, feature_cols, encoders


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, X, y, feature_cols, encoders = preprocess_pipeline()
    print("Sample cleaned data:")
    print(df[["player_name", "opponent_team", "venue",
               "recent_avg_runs", "runs_scored"]].head(10).to_string())
