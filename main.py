"""
main.py
-------
ONE-CLICK ENTRY POINT

Run this file to execute the entire pipeline:
  generate → preprocess → train → visualize → predict
"""

import os, sys

def banner(title):
    print("\n" + "█"*60)
    print(f"  {title}")
    print("█"*60 + "\n")


def main():
    banner("🏏  CRICKET PLAYER RUNS PREDICTION SYSTEM  🏏")

    # ── Step 0: Generate dataset ───────────────────────────────────────────
    banner("STEP 0 · GENERATING DATASET")
    import generate_dataset   # runs on import

    # ── Step 1: Train model ────────────────────────────────────────────────
    banner("STEP 1 · PREPROCESSING + TRAINING")
    from train_model import train_pipeline
    sk_model, scratch_model, X_test, y_test, y_pred_sk, df = train_pipeline()

    # ── Step 2: Visualizations ─────────────────────────────────────────────
    banner("STEP 2 · GENERATING VISUALIZATIONS")
    from visualization import generate_all_plots
    generate_all_plots(df, y_test, y_pred_sk)

    # ── Step 3: Sample predictions ─────────────────────────────────────────
    banner("STEP 3 · SAMPLE PREDICTIONS")
    from predict import batch_predict, load_model
    bundle = load_model()

    scenarios = [
        {"player_name": "Virat Kohli",    "opponent_team": "Mumbai Indians",
         "venue": "M. Chinnaswamy Stadium, Bangalore",       "recent_avg": 52.0},
        {"player_name": "Rohit Sharma",   "opponent_team": "Chennai Super Kings",
         "venue": "Wankhede Stadium, Mumbai",                "recent_avg": 38.0},
        {"player_name": "MS Dhoni",       "opponent_team": "Delhi Capitals",
         "venue": "M.A. Chidambaram Stadium, Chennai",       "recent_avg": 28.0},
        {"player_name": "Jos Buttler",    "opponent_team": "Kolkata Knight Riders",
         "venue": "Sawai Mansingh Stadium, Jaipur",          "recent_avg": 60.0},
        {"player_name": "Glenn Maxwell",  "opponent_team": "Sunrisers Hyderabad",
         "venue": "M. Chinnaswamy Stadium, Bangalore",       "recent_avg": 35.0},
    ]

    preds = batch_predict(scenarios, bundle)
    print(f"{'Player':<22} {'Opponent':<30} {'Form':>5}  {'Pred':>6}")
    print("─" * 70)
    for p in preds:
        print(f"{p['player_name']:<22} {p['opponent_team']:<30} "
              f"{p['recent_avg']:>5.0f}  {p['predicted_runs']:>6.0f} runs")

    banner("✅  ALL DONE — check the 'outputs/' folder for charts!")


if __name__ == "__main__":
    main()
