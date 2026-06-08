"""
predict.py
----------
PREDICTION SYSTEM — predict runs for any player/opponent/venue combo.

🏏 Cricket Analogy:
The scout opens the model (like a well-kept coaching notebook),
feeds in today's match conditions, and gets back an estimated score.
"Rohit Sharma vs MI at Wankhede, current form 42 → predicted 47 runs."
"""

import pickle
import numpy as np
import os


# ──────────────────────────────────────────────────────────────────────────────
def load_model(path: str = "model.pkl") -> dict:
    """Load the trained model bundle from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Model file '{path}' not found. Run train_model.py first."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print("✅ Model loaded successfully.")
    return bundle


# ──────────────────────────────────────────────────────────────────────────────
def encode_input(
    player_name: str,
    opponent_team: str,
    venue: str,
    recent_avg: float,
    encoders: dict,
) -> np.ndarray:
    """
    Encode a single prediction request into the same numeric format
    used during training.

    If an unseen label is passed, we fall back to the closest known
    label (index 0) — like a selector defaulting to 'average' for a
    new, unrated player.
    """

    def safe_encode(le, value):
        if value in le.classes_:
            return le.transform([value])[0]
        # Unseen label — warn and use index 0
        print(f"   ⚠️  '{value}' not in training data; using fallback encoding.")
        return 0

    p_enc  = safe_encode(encoders["player_name"],   player_name)
    o_enc  = safe_encode(encoders["opponent_team"], opponent_team)
    v_enc  = safe_encode(encoders["venue"],         venue)

    return np.array([[p_enc, o_enc, v_enc, recent_avg]])


# ──────────────────────────────────────────────────────────────────────────────
def predict_runs(
    player_name: str,
    opponent_team: str,
    venue: str,
    recent_avg: float,
    bundle: dict = None,
) -> float:
    """
    Predict runs for one match scenario.

    🏏 Analogy:
    Feed the model a matchup — 'Virat Kohli vs Mumbai Indians at Wankhede,
    recent average 52' — and it consults its learned patterns to give
    a best-guess score.

    Returns
    -------
    predicted_runs : float
    """
    if bundle is None:
        bundle = load_model()

    model    = bundle["model"]
    encoders = bundle["encoders"]

    X_input  = encode_input(player_name, opponent_team, venue, recent_avg, encoders)
    pred     = model.predict(X_input)[0]
    # Runs can't be negative
    return max(0.0, round(float(pred), 1))


# ──────────────────────────────────────────────────────────────────────────────
def interactive_predict():
    """
    Command-line interactive prediction session.
    Lets you type a player name, opponent, venue, and recent average
    to get a predicted score — just like a selector running a quick
    pre-match analysis.
    """
    print("\n" + "="*60)
    print("  🏏  CRICKET RUNS PREDICTION — INTERACTIVE MODE")
    print("="*60)

    bundle   = load_model()
    encoders = bundle["encoders"]

    players   = list(encoders["player_name"].classes_)
    opponents = list(encoders["opponent_team"].classes_)
    venues    = list(encoders["venue"].classes_)

    print("\n📋 Known Players:")
    for i, p in enumerate(players, 1):
        print(f"   {i:2d}. {p}")

    print("\n📋 Known Teams:")
    for i, t in enumerate(opponents, 1):
        print(f"   {i:2d}. {t}")

    print("\n📋 Known Venues:")
    for i, v in enumerate(venues, 1):
        print(f"   {i:2d}. {v}")

    print("\n" + "-"*60)
    print("Enter match details (or press Ctrl+C to quit):\n")

    while True:
        try:
            player  = input("🧑 Player Name      : ").strip()
            opp     = input("🏟️  Opponent Team    : ").strip()
            venue   = input("📍 Venue             : ").strip()
            recent  = float(input("📈 Recent Avg Runs  : "))

            runs = predict_runs(player, opp, venue, recent, bundle)

            print(f"\n{'─'*50}")
            print(f"  🏏 Predicted Runs for {player}")
            print(f"  vs {opp} at {venue.split(',')[0]}")
            print(f"  Recent Form  : {recent} runs/match")
            print(f"  ➡️  Predicted : {runs:.0f} runs")
            print(f"{'─'*50}\n")

        except ValueError:
            print("❌ Invalid input. Please enter a numeric recent average.\n")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting prediction system. Good luck!")
            break


# ──────────────────────────────────────────────────────────────────────────────
def batch_predict(scenarios: list, bundle: dict = None) -> list:
    """
    Predict runs for a list of match scenarios.

    Parameters
    ----------
    scenarios : list of dicts with keys:
        player_name, opponent_team, venue, recent_avg
    bundle : pre-loaded model bundle (optional)

    Returns
    -------
    list of predicted run values
    """
    if bundle is None:
        bundle = load_model()

    results = []
    for s in scenarios:
        pred = predict_runs(
            s["player_name"],
            s["opponent_team"],
            s["venue"],
            s["recent_avg"],
            bundle,
        )
        results.append({**s, "predicted_runs": pred})
    return results


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Demo batch predictions ─────────────────────────────────────────────
    print("\n🏏  SAMPLE BATCH PREDICTIONS\n")
    bundle = load_model()

    sample_scenarios = [
        {"player_name": "Virat Kohli",    "opponent_team": "Mumbai Indians",
         "venue": "M. Chinnaswamy Stadium, Bangalore",       "recent_avg": 52.0},
        {"player_name": "Rohit Sharma",   "opponent_team": "Chennai Super Kings",
         "venue": "Wankhede Stadium, Mumbai",                "recent_avg": 38.0},
        {"player_name": "MS Dhoni",       "opponent_team": "Delhi Capitals",
         "venue": "M.A. Chidambaram Stadium, Chennai",       "recent_avg": 28.0},
        {"player_name": "Jos Buttler",    "opponent_team": "Kolkata Knight Riders",
         "venue": "Sawai Mansingh Stadium, Jaipur",          "recent_avg": 60.0},
        {"player_name": "KL Rahul",       "opponent_team": "Royal Challengers Bangalore",
         "venue": "IS Bindra Stadium, Mohali",               "recent_avg": 44.0},
    ]

    preds = batch_predict(sample_scenarios, bundle)
    print(f"{'Player':<22} {'Opponent':<30} {'RecAvg':>7}  {'Predicted':>10}")
    print("─" * 75)
    for p in preds:
        print(f"{p['player_name']:<22} {p['opponent_team']:<30} "
              f"{p['recent_avg']:>7.1f}  {p['predicted_runs']:>10.0f} runs")

    # ── Interactive mode ───────────────────────────────────────────────────
    print("\n")
    try:
        go = input("🎮 Launch interactive prediction? (y/n): ").strip().lower()
        if go == "y":
            interactive_predict()
    except KeyboardInterrupt:
        pass
