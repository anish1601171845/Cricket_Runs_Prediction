"""
train_model.py
--------------
MODEL BUILDING & EVALUATION

🏏 Cricket Analogy:
Linear Regression is like a veteran analyst who watches every
Virat Kohli innings and says: "On batting-friendly pitches,
against weak bowling attacks, with good recent form, Kohli
typically scores around 55." The model learns those relationships
from historical scorecards and then generalises to new matches.

Two implementations:
  A) Linear Regression FROM SCRATCH (pure NumPy)
     — so you understand every equation.
  B) Scikit-Learn LinearRegression
     — production-grade, for comparison.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import preprocess_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# A  ·  LINEAR REGRESSION FROM SCRATCH
# ══════════════════════════════════════════════════════════════════════════════

class LinearRegressionScratch:
    """
    Linear Regression implemented using the Normal Equation:

        θ = (XᵀX)⁻¹ Xᵀy

    🏏 Analogy:
    Imagine plotting runs on a graph for different 'recent averages'.
    Drawing the best-fit line through those dots gives you the
    regression line. The Normal Equation finds that line mathematically
    in one shot — no guesswork needed.
    """

    def __init__(self):
        self.theta = None          # Model weights (coefficients + intercept)
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train: solve the Normal Equation.
        Add a bias column (all 1s) so the intercept is learned too.
        """
        # Add intercept column
        ones  = np.ones((X.shape[0], 1))
        X_b   = np.hstack([ones, X])          # shape (n, features+1)

        # θ = (XᵀX)⁻¹ Xᵀy  — the closed-form solution
        XtX   = X_b.T @ X_b
        Xty   = X_b.T @ y
        self.theta = np.linalg.pinv(XtX) @ Xty   # pinv handles near-singular matrices
        print("✅ [Scratch] Model trained via Normal Equation")
        print(f"   Intercept : {self.theta[0]:.4f}")
        print(f"   Coefficients: {self.theta[1:]}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        X_b  = np.hstack([ones, X])
        return X_b @ self.theta


# ══════════════════════════════════════════════════════════════════════════════
# B  ·  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model_name: str, y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, R².

    🏏 Analogy:
    MAE  → on average, how many runs off is our prediction?
           (off by 10 runs = "decent selector's intuition")
    RMSE → penalises big misses harder.
           (being 50 runs off is much worse than 10 runs off twice)
    R²   → how much of the run-variation does our model explain?
           (1.0 = perfect; 0.0 = as good as just guessing the mean)
    """
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n📊 [{model_name}] Evaluation Metrics:")
    print(f"   MAE  (Mean Absolute Error) : {mae:.4f}  runs")
    print(f"   RMSE (Root Mean Sq. Error) : {rmse:.4f} runs")
    print(f"   R²   (Coefficient of Det.) : {r2:.4f}")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ══════════════════════════════════════════════════════════════════════════════
# C  ·  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_pipeline():
    print("\n" + "="*60)
    print("  🏏  CRICKET RUNS PREDICTION — MODEL TRAINING")
    print("="*60 + "\n")

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    df, X, y, feature_cols, encoders = preprocess_pipeline()

    # ── 2. Train / Test Split (80 / 20) ───────────────────────────────────────
    # 🏏 Analogy: We train on IPL seasons 2015–2021 and test on 2022–2023.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📂 Train set : {X_train.shape[0]:,} records")
    print(f"   Test  set : {X_test.shape[0]:,} records")

    results = []

    # ── 3a. Train — Scratch Implementation ────────────────────────────────────
    print("\n🔨 Training Linear Regression (FROM SCRATCH)…")
    scratch_model = LinearRegressionScratch()
    scratch_model.fit(X_train, y_train)
    y_pred_scratch = scratch_model.predict(X_test)
    results.append(evaluate("Scratch Linear Regression", y_test, y_pred_scratch))

    # ── 3b. Train — Scikit-Learn ──────────────────────────────────────────────
    print("\n🔨 Training Linear Regression (SCIKIT-LEARN)…")
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    y_pred_sk = sk_model.predict(X_test)
    results.append(evaluate("Scikit-Learn LinearRegression", y_test, y_pred_sk))

    print("\n🔍 Sklearn Coefficients:")
    for fname, coef in zip(feature_cols, sk_model.coef_):
        print(f"   {fname:25s} : {coef:+.4f}")
    print(f"   {'intercept':25s} : {sk_model.intercept_:+.4f}")

    # ── 4. Comparison table ───────────────────────────────────────────────────
    print("\n📋 Model Comparison:")
    comp = pd.DataFrame(results).set_index("model")
    print(comp.round(4).to_string())

    # ── 5. Save best model (sklearn) + metadata ───────────────────────────────
    model_data = {
        "model":        sk_model,
        "encoders":     encoders,
        "feature_cols": feature_cols,
        "df_sample":    df[["player_name", "opponent_team", "venue",
                             "recent_avg_runs", "runs_scored"]].head(5),
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("\n✅ Trained model saved → model.pkl")

    # ── 6. Save predictions for visualisation ────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    pred_df = pd.DataFrame({
        "actual":    y_test,
        "predicted": y_pred_sk.round(1),
    })
    pred_df.to_csv("outputs/predictions.csv", index=False)
    print("✅ Predictions saved → outputs/predictions.csv")

    return sk_model, scratch_model, X_test, y_test, y_pred_sk, df


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_pipeline()
