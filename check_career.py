
import pickle
import pandas as pd
from pathlib import Path

DATA_FILE = "nba_data.pkl"

def check_career_games():
    try:
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
        
        career = data.get("career_basic", pd.DataFrame())
        if career.empty:
            print("No career stats found.")
            return

        # Filter for current season
        current = career[career["SEASON_ID"] == "2025-26"]
        if current.empty:
             print("No 2025-26 career entry found.")
        else:
             print(f"2025-26 Games Played (GP): {current.iloc[0]['GP']}")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_career_games()
