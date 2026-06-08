"""
visualization.py
----------------
ALL CHARTS & GRAPHS FOR THE PROJECT

🏏 Cricket Analogy:
Just as commentators use wagon-wheels and Manhattan graphs to tell
the story of an innings, we use matplotlib charts to narrate what
our data and model are doing.

Charts produced:
  1. Top Run Scorers (Bar Chart)
  2. Runs Distribution (Histogram)
  3. Venue-wise Average Runs (Bar Chart)
  4. Opponent-wise Average Runs (Bar Chart)
  5. Correlation Heatmap
  6. Actual vs Predicted Scatter Plot
  7. Residuals Distribution
  8. Recent Average vs Runs Scored (Scatter)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — works in any env
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ─── Consistent style ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = sns.color_palette("crest", 12)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
def save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1 · Top Run Scorers
# ══════════════════════════════════════════════════════════════════════════════
def plot_top_scorers(df: pd.DataFrame, top_n: int = 10):
    """
    Bar chart of players with the highest total runs.
    🏏 Like the IPL Orange Cap leaderboard.
    """
    top = (
        df.groupby("player_name")["runs_scored"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top["player_name"][::-1], top["runs_scored"][::-1],
                   color=COLORS[:top_n][::-1], edgecolor="white")
    ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=9)
    ax.set_title(f"🏆 Top {top_n} Run Scorers (Career Total)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Runs")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save(fig, "01_top_run_scorers.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2 · Runs Distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_runs_distribution(df: pd.DataFrame):
    """
    Histogram of runs per innings.
    🏏 Most batters score low; centuries are rare — classic right skew.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["runs_scored"], bins=40, color=COLORS[2], edgecolor="white", alpha=0.85)
    ax.axvline(df["runs_scored"].mean(),   color="crimson", lw=1.8, ls="--", label=f"Mean  {df['runs_scored'].mean():.1f}")
    ax.axvline(df["runs_scored"].median(), color="navy",    lw=1.8, ls=":",  label=f"Median {df['runs_scored'].median():.0f}")
    ax.set_title("📊 Distribution of Runs Scored per Innings", fontsize=14, fontweight="bold")
    ax.set_xlabel("Runs Scored")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    save(fig, "02_runs_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3 · Venue-wise Average Runs
# ══════════════════════════════════════════════════════════════════════════════
def plot_venue_performance(df: pd.DataFrame):
    """
    Average runs per venue — shows batting vs bowling friendly pitches.
    🏏 Chinnaswamy (Bangalore) is famously a run-fest; Chennai is slower.
    """
    venue_avg = (
        df.groupby("venue")["runs_scored"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    venue_avg["venue_short"] = venue_avg["venue"].str.split(",").str[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(venue_avg["venue_short"], venue_avg["runs_scored"],
                  color=COLORS, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title("🏟️  Average Runs by Venue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Venue")
    ax.set_ylabel("Avg Runs")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    save(fig, "03_venue_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Opponent-wise Average Runs
# ══════════════════════════════════════════════════════════════════════════════
def plot_opponent_performance(df: pd.DataFrame):
    """
    Average runs scored against each team — reveals weak bowling attacks.
    🏏 Some teams give away more runs; others are tighter.
    """
    opp_avg = (
        df.groupby("opponent_team")["runs_scored"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(opp_avg["opponent_team"], opp_avg["runs_scored"],
                  color=sns.color_palette("rocket", len(opp_avg)), edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title("⚔️  Average Runs vs Each Opponent", fontsize=14, fontweight="bold")
    ax.set_xlabel("Opponent Team")
    ax.set_ylabel("Avg Runs Conceded")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    save(fig, "04_opponent_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5 · Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Heatmap of numeric feature correlations.
    🏏 High correlation between strike_rate and runs? Of course —
    faster scorers accumulate runs quicker, all else equal.
    """
    num_df = df[["runs_scored", "balls_faced", "strike_rate",
                 "fours", "sixes", "recent_avg_runs"]].copy()
    num_df.columns = ["Runs", "Balls", "Strike Rate", "4s", "6s", "Recent Avg"]

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, ax=ax, linewidths=0.5, square=True,
                annot_kws={"size": 10})
    ax.set_title("🔥 Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6 · Actual vs Predicted
# ══════════════════════════════════════════════════════════════════════════════
def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Scatter plot comparing actual vs predicted runs.
    🏏 Perfect predictions → all dots on the diagonal.
    Spread = how far off we are on individual innings.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color=COLORS[5], edgecolors="none")

    lim = max(y_test.max(), y_pred.max()) + 5
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect Prediction")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title("🎯 Actual vs Predicted Runs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Runs")
    ax.set_ylabel("Predicted Runs")
    ax.legend()
    fig.tight_layout()
    save(fig, "06_actual_vs_predicted.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7 · Residuals Distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Histogram of prediction errors (residuals).
    🏏 Centred at 0 → model is unbiased. Wide spread → high uncertainty
    (cricket scores ARE volatile — even the best model has limits).
    """
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=40, color=COLORS[8], edgecolor="white", alpha=0.85)
    ax.axvline(0, color="crimson", lw=1.8, ls="--", label="Zero Error")
    ax.set_title("📉 Prediction Residuals (Error Distribution)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    save(fig, "07_residuals.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8 · Recent Average vs Runs Scored
# ══════════════════════════════════════════════════════════════════════════════
def plot_recent_avg_vs_runs(df: pd.DataFrame):
    """
    Scatter of recent_avg_runs vs runs_scored.
    🏏 A player in form (high recent average) tends to score more.
    This is the most important feature in our model.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["recent_avg_runs"], df["runs_scored"],
               alpha=0.25, s=12, color=COLORS[3])

    # Trend line
    m, b = np.polyfit(df["recent_avg_runs"], df["runs_scored"], 1)
    xs = np.linspace(df["recent_avg_runs"].min(), df["recent_avg_runs"].max(), 100)
    ax.plot(xs, m * xs + b, color="crimson", lw=2, label=f"Trend (slope={m:.2f})")

    ax.set_title("📈 Recent Form vs Runs Scored", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recent Average Runs (Last 5 Innings)")
    ax.set_ylabel("Runs Scored")
    ax.legend()
    fig.tight_layout()
    save(fig, "08_recent_avg_vs_runs.png")


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def generate_all_plots(df: pd.DataFrame, y_test: np.ndarray = None,
                       y_pred: np.ndarray = None):
    print("\n" + "="*60)
    print("  🏏  CRICKET RUNS PREDICTION — VISUALIZATIONS")
    print("="*60 + "\n")

    print("📊 Plot 1: Top Run Scorers")
    plot_top_scorers(df)

    print("📊 Plot 2: Runs Distribution")
    plot_runs_distribution(df)

    print("📊 Plot 3: Venue Performance")
    plot_venue_performance(df)

    print("📊 Plot 4: Opponent Performance")
    plot_opponent_performance(df)

    print("📊 Plot 5: Correlation Heatmap")
    plot_correlation_heatmap(df)

    if y_test is not None and y_pred is not None:
        print("📊 Plot 6: Actual vs Predicted")
        plot_actual_vs_predicted(y_test, y_pred)

        print("📊 Plot 7: Residuals")
        plot_residuals(y_test, y_pred)

    print("📊 Plot 8: Recent Avg vs Runs")
    plot_recent_avg_vs_runs(df)

    print(f"\n✅ All plots saved to '{OUT_DIR}/' folder.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessing import preprocess_pipeline
    df, X, y, _, _ = preprocess_pipeline()

    # Load predictions if available
    try:
        import pandas as pd_inner
        preds_df = pd_inner.read_csv("outputs/predictions.csv")
        y_test = preds_df["actual"].values
        y_pred = preds_df["predicted"].values
    except FileNotFoundError:
        print("⚠️  No predictions.csv found; skipping actual-vs-predicted plots.")
        y_test = y_pred = None

    generate_all_plots(df, y_test, y_pred)
