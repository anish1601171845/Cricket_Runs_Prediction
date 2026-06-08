"""
generate_dataset.py
-------------------
Generates a realistic IPL-style batting dataset for demonstration.
Think of this as our "scorebook" — every entry is like a scorecard entry
from a past match, recording how each batsman performed.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ─── Players & their skill profiles ───────────────────────────────────────────
# Just like how Virat Kohli averages ~50 while a lower-order bat averages ~15,
# each player has a base average and variance.
PLAYERS = {
    "Virat Kohli":       {"avg": 45, "std": 22, "sr_base": 135},
    "Rohit Sharma":      {"avg": 42, "std": 25, "sr_base": 130},
    "MS Dhoni":          {"avg": 35, "std": 20, "sr_base": 140},
    "KL Rahul":          {"avg": 40, "std": 22, "sr_base": 132},
    "Shubman Gill":      {"avg": 38, "std": 21, "sr_base": 128},
    "David Warner":      {"avg": 43, "std": 24, "sr_base": 138},
    "AB de Villiers":    {"avg": 48, "std": 26, "sr_base": 158},
    "Hardik Pandya":     {"avg": 28, "std": 18, "sr_base": 145},
    "Suresh Raina":      {"avg": 32, "std": 20, "sr_base": 135},
    "Rishabh Pant":      {"avg": 34, "std": 23, "sr_base": 148},
    "Quinton de Kock":   {"avg": 36, "std": 22, "sr_base": 133},
    "Ishan Kishan":      {"avg": 30, "std": 21, "sr_base": 136},
    "Faf du Plessis":    {"avg": 37, "std": 20, "sr_base": 131},
    "Jos Buttler":       {"avg": 44, "std": 25, "sr_base": 150},
    "Glenn Maxwell":     {"avg": 33, "std": 28, "sr_base": 160},
    "Andre Russell":     {"avg": 30, "std": 22, "sr_base": 175},
    "Kieron Pollard":    {"avg": 28, "std": 19, "sr_base": 155},
    "Sanju Samson":      {"avg": 33, "std": 23, "sr_base": 138},
    "Dinesh Karthik":    {"avg": 25, "std": 17, "sr_base": 145},
    "Ambati Rayudu":     {"avg": 30, "std": 19, "sr_base": 130},
}

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad",
]

VENUES = [
    "Wankhede Stadium, Mumbai",
    "M.A. Chidambaram Stadium, Chennai",
    "M. Chinnaswamy Stadium, Bangalore",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "IS Bindra Stadium, Mohali",
    "Sawai Mansingh Stadium, Jaipur",
    "Rajiv Gandhi International Stadium, Hyderabad",
    "Maharashtra Cricket Association Stadium, Pune",
    "DY Patil Stadium, Mumbai",
]

# Venue pitch factors: some grounds are batting-friendly (>1.0), some aren't
VENUE_FACTOR = {v: round(np.random.uniform(0.85, 1.20), 2) for v in VENUES}
# Opponent bowling strength: harder opponents → lower scores
TEAM_BOWLING = {t: round(np.random.uniform(0.85, 1.15), 2) for t in TEAMS}

records = []
dates = pd.date_range("2015-04-01", "2023-05-30", freq="3D")

for _ in range(2000):
    player = np.random.choice(list(PLAYERS.keys()))
    p = PLAYERS[player]

    opponent = np.random.choice(TEAMS)
    venue    = np.random.choice(VENUES)
    date     = pd.Timestamp(np.random.choice(dates))

    # Base runs: player avg × venue factor × opponent bowling factor
    base = p["avg"] * VENUE_FACTOR[venue] * TEAM_BOWLING[opponent]
    runs = int(max(0, np.random.normal(base, p["std"])))

    balls = max(1, int(runs / (p["sr_base"] / 100) + np.random.randint(-5, 6)))
    sr    = round((runs / balls) * 100, 2)
    fours = int(runs * np.random.uniform(0.05, 0.15))
    sixes = int(runs * np.random.uniform(0.02, 0.10))

    records.append({
        "player_name":   player,
        "match_date":    date.strftime("%Y-%m-%d"),
        "opponent_team": opponent,
        "venue":         venue,
        "runs_scored":   runs,
        "balls_faced":   balls,
        "strike_rate":   sr,
        "fours":         fours,
        "sixes":         sixes,
    })

df = pd.DataFrame(records).sort_values("match_date").reset_index(drop=True)
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/cricket_data.csv", index=False)
print(f"✅ Dataset generated: {len(df)} records → dataset/cricket_data.csv")
print(df.head())
