import sys
import io
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check(name, val):
    print(f"\n--- Testing {name} ---")
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2024-25',
            measure_type_detailed_defense=val,
            per_mode_detailed='Totals',
            timeout=30
        ).get_data_frames()[0]
        
        cols = list(df.columns)
        print(f"Columns: {cols[:5]} ... Total {len(cols)}")
        
        if "AND_ONES" in cols: print("!!! FOUND AND_ONES !!!")
        if "USG_PCT" in cols: print("!!! FOUND USG_PCT !!!")
        if "DEF_RATING" in cols: print("!!! FOUND DEF_RATING !!!")
        
        # Check if it looks like Base
        if "PTS" in cols and "REB" in cols and "AST" in cols:
            print("Looks like BASE stats.")
            
    except Exception as e:
        print(f"Error: {e}")

# 1. Test Advanced (Should have USG_PCT)
check("Advanced", "Advanced")

# 2. Test Misc (Should have AND_ONES)
check("Misc", "Misc")

# 3. Test Usage 
check("Usage", "Usage")

# 4. Test Scoring
check("Scoring", "Scoring")
