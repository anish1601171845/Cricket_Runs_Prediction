# 🏏 Cricket Player Runs Prediction System

> *"Linear Regression is like estimating Virat Kohli's next score by observing
> his previous performances under similar conditions — venue, opponent, and
> recent form all factor in."*

---

## 📋 Project Overview

A complete end-to-end Machine Learning project that predicts how many runs a
batsman will score in an upcoming IPL match.

We implement **Linear Regression** two ways — from scratch using NumPy and via
Scikit-Learn — and compare both on the same dataset.

---

## 📁 Project Structure

```
Cricket_Runs_Prediction/
│
├── dataset/
│   └── cricket_data.csv          ← auto-generated IPL-style batting records
│
├── generate_dataset.py           ← builds a realistic 2000-row dataset
├── preprocessing.py              ← cleaning, encoding, feature engineering
├── train_model.py                ← trains Scratch + Sklearn models, saves model.pkl
├── predict.py                    ← loads model, runs batch & interactive predictions
├── visualization.py              ← all 8 charts saved to outputs/
├── main.py                       ← ONE-CLICK entry point (runs everything)
│
├── model.pkl                     ← saved trained model (auto-created)
├── encoders.pkl                  ← saved label encoders (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project folder
cd Cricket_Runs_Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything in one shot
python main.py
```

Or run individual modules:

```bash
python generate_dataset.py   # step 0 — create dataset
python preprocessing.py      # step 1 — clean & encode
python train_model.py        # step 2 — train & evaluate
python visualization.py      # step 3 — generate all charts
python predict.py            # step 4 — sample + interactive predictions
```

---

## 🏏 Cricket Analogies (How Each Step Works)

| ML Concept | Cricket Analogy |
|---|---|
| **Dataset** | The scorebook — every innings ever recorded |
| **Features** | Match conditions: who's batting, who's bowling, where, what form |
| **Target** | The final score on the scoreboard |
| **Train/Test Split** | Train on IPL 2015–2021, test on 2022–2023 |
| **Linear Regression** | Best-fit line through the scatter of past scores |
| **Normal Equation** | Finding the best-fit line mathematically, in one go |
| **MAE** | On average, how many runs off is our prediction? |
| **RMSE** | Penalises big misses more — a 50-run miss is far worse than 10 |
| **R² Score** | How much of the score variation does our model explain? |
| **recent_avg_runs** | The selector's "form" check — last 5 innings average |
| **Label Encoding** | Converting "Virat Kohli" → 5 (model speaks numbers) |
| **Residuals** | How far off was our prediction? Centred at 0 = unbiased |

---

## 🔢 Input Features

| Feature | Description |
|---|---|
| `player_name_enc` | Label-encoded player identity |
| `opponent_team_enc` | Label-encoded opposing team |
| `venue_enc` | Label-encoded match ground |
| `recent_avg_runs` | Rolling 5-innings average (player form) |

**Target:** `runs_scored`

---

## 📊 Visualizations (in `outputs/`)

| File | Chart |
|---|---|
| `01_top_run_scorers.png` | Horizontal bar — career run totals |
| `02_runs_distribution.png` | Histogram — runs per innings |
| `03_venue_performance.png` | Bar — avg runs per ground |
| `04_opponent_performance.png` | Bar — avg runs vs each team |
| `05_correlation_heatmap.png` | Heatmap — numeric correlations |
| `06_actual_vs_predicted.png` | Scatter — actual vs model output |
| `07_residuals.png` | Histogram — prediction errors |
| `08_recent_avg_vs_runs.png` | Scatter + trend line — form vs output |

---

## 📈 Model Evaluation Results

```
Model                         MAE      RMSE     R²
──────────────────────────────────────────────────
Scratch Linear Regression    ~18 runs  ~22 runs  ~0.034
Scikit-Learn LinearRegression ~18 runs ~22 runs  ~0.034
```

> **Why low R²?** Cricket scores are inherently volatile — even the best
> batting form can end on a golden duck. More features (weather, toss result,
> pitch report, bowling matchup stats) would improve this further.

---

## 🎯 Sample Predictions

```
Player                 Opponent                  Form   Predicted
─────────────────────────────────────────────────────────────────
Virat Kohli            Mumbai Indians              52       42 runs
Rohit Sharma           Chennai Super Kings         38       38 runs
MS Dhoni               Delhi Capitals              28       34 runs
Jos Buttler            Kolkata Knight Riders        60       43 runs
Glenn Maxwell          Sunrisers Hyderabad          35       34 runs
```

---

## 💼 Resume Description

> **Cricket Player Runs Prediction System** | Python · Scikit-Learn · Pandas · Matplotlib
>
> Built an end-to-end ML regression pipeline to predict IPL batsmen's match
> scores. Engineered a rolling 5-match form feature, label-encoded categorical
> variables (player/venue/opponent), and implemented Linear Regression both
> from scratch (Normal Equation) and via Scikit-Learn. Evaluated using MAE,
> RMSE, and R². Produced 8 analytical visualisations including correlation
> heatmaps and residual plots.

---

## 🎤 Interview Questions & Answers

**Q1. What is Linear Regression?**
A model that fits a straight line through data points by minimising the sum of
squared errors. The Normal Equation solves θ = (XᵀX)⁻¹Xᵀy in closed form.

**Q2. Why did you use rolling average as a feature?**
A player's recent form is a better predictor of next-match score than career
average. A batter in peak form (last 5 avg = 60) is more likely to score than
one whose long-run average is 60 but recent form is 20.

**Q3. Why is R² low here?**
Cricket scores are high-variance — even in-form players get dismissed early.
The model captures trend but can't account for unstructured randomness (lucky
wickets, weather changes). More features would improve it.

**Q4. Scratch vs Sklearn — any difference in results?**
Both produce identical results because both solve the same Normal Equation.
Sklearn adds numerical stability improvements (SVD-based solver) but for our
clean dataset both match to several decimal places.

**Q5. How would you improve this model?**
- Add more features: toss result, pitch type, bowler quality index, weather
- Use non-linear models: Random Forest, XGBoost
- Per-player models with more data
- Time-series cross-validation to respect temporal ordering

**Q6. What is Label Encoding? Could you use One-Hot instead?**
Label Encoding maps categories to integers (Kohli→5). It introduces false
ordinal relationships. One-Hot Encoding creates binary columns and avoids
this — better for tree models. For linear regression, one-hot is generally
preferred, but with many categories it creates wide matrices.

**Q7. What does MAE vs RMSE tell you?**
MAE treats all errors equally. RMSE squares errors first, so large mistakes
(predicting 100 when actual is 10) are penalised much more heavily. In
cricket prediction, RMSE better reflects that big misses are costly.
