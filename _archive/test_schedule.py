from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import pandas as pd

try:
    # Use yesterday/today string
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Checking Scoreboard for {today_str}...")
    
    board = scoreboardv2.ScoreboardV2(game_date=today_str)
    games = board.game_header.get_data_frame()
    
    print("\nColumns:", games.columns.tolist())
    
    # Filter for Portland (ID: 1610612757)
    blazers = games[games['HOME_TEAM_ID'] == 1610612757]
    if blazers.empty:
        blazers = games[games['VISITOR_TEAM_ID'] == 1610612757]
        
    if not blazers.empty:
        print("\nBlazers Game Found:")
        print(blazers.iloc[0].to_dict())
    else:
        print("\nNo Blazers game today.")

except Exception as e:
    print(f"Error: {e}")
