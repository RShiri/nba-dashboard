import sys
import io
from nba_api.stats.endpoints import leaguedashplayerstats, leaguehustlestatsplayer
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("--- Inspecting Scoring Columns ---")
try:
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season='2024-25',
        measure_type_detailed_defense='Scoring',
        per_mode_detailed='Totals',
        timeout=30
    ).get_data_frames()[0]
    print(list(df.columns))
except Exception as e:
    print(f"Error Scoring: {e}")

print("\n--- Inspecting Hustle Columns ---")
try:
    df = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
        season='2024-25',
        per_mode_time='Totals',
        timeout=30
    ).get_data_frames()[0]
    print(list(df.columns))
except Exception as e:
    print(f"Error Hustle: {e}")
