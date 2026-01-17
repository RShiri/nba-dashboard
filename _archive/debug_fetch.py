import pickle
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2

DATA_FILE = "nba_data.pkl"

def debug_update():
    print(f"Current System Time: {datetime.now()}")
    
    try:
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
            
        logs = data.get("game_logs_2025_26", pd.DataFrame())
        
        if logs.empty:
            print("Logs are empty!")
            return

        # Check Dtypes and Max Date
        print("\n--- Local Data State ---")
        print(logs.dtypes)
        if "GAME_DATE" in logs.columns:
            # Force conversion just to be sure what we have
            dates = pd.to_datetime(logs["GAME_DATE"])
            max_date = dates.max()
            print(f"\nMax Date in logs: {max_date}")
            print(f"Tail of logs:\n{logs.sort_values('GAME_DATE').tail(3)[['GAME_DATE', 'MATCHUP', 'WL']]}")
            
            # Simulate date check
            start_date = max_date + timedelta(days=1)
            print(f"\n--- Simulation ---")
            print(f"Simulating check starting from: {start_date}")
            
            # Test one known date where a game SHOULD represent
            test_date = "2025-12-26" # Random guess for a game date? Or lets just try 'tomorrow' relative to last game
            
            # Actually lets just run the scoreboard check for the very next day after max_date
            check_scoreboard(start_date)
            
        else:
            print("No GAME_DATE column found!")

    except Exception as e:
        print(f"Error reading pickle: {e}")

def check_scoreboard(date_obj):
    d_str = date_obj.strftime("%Y-%m-%d")
    print(f"Checking Scoreboard for {d_str}...")
    try:
        board = scoreboardv2.ScoreboardV2(game_date=d_str).game_header.get_data_frame()
        print(f"Board shape: {board.shape}")
        if not board.empty:
            print("Sample row:", board.iloc[0].to_dict())
            
            # Check for Blazers (1610612757)
            team_id = 1610612757
            game = board[(board['HOME_TEAM_ID'] == team_id) | (board['VISITOR_TEAM_ID'] == team_id)]
            if not game.empty:
                print("Found Blazers game:", game.iloc[0].to_dict())
            else:
                print("No Blazers game found on this date.")
        else:
            print("Board is empty.")
            
    except Exception as e:
        print(f"Error checking scoreboard: {e}")

if __name__ == "__main__":
    debug_update()
