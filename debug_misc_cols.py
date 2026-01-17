import sys
import io
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("--- Inspecting Misc Columns ---")
try:
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season='2024-25',
        measure_type_detailed_defense='Misc',
        per_mode_detailed='Totals',
        timeout=30
    ).get_data_frames()[0]
    
    print(list(df.columns))
    
except Exception as e:
    print(f"Error: {e}")
