import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptstats
import sys

# Force UTF-8 (best effort) or just avoid emojis
sys.stdout.reconfigure(encoding='utf-8')

def check_endpoint(measure_type):
    print(f"\n--- Checking MeasureType: {measure_type} ---")
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            measure_type_detailed_defense=measure_type,
            per_mode_detailed='Totals',
            timeout=30
        ).get_data_frames()[0]
        
        cols = list(df.columns)
        print(f"Columns ({len(cols)}): {cols}")
        
        matches = [c for c in cols if "AND" in c.upper() or "1" in c] # "1" might find "And1"
        if matches:
            print(f"Potential Matches: {matches}")
        else:
            print("No fuzzy matches found.")
            
    except Exception as e:
        print(f"Error: {e}")

def check_drives():
    print(f"\n--- Checking PT MeasureType: Drives ---")
    try:
        df = leaguedashptstats.LeagueDashPtStats(
            season="2025-26",
            pt_measure_type='Drives',
            player_or_team='Player',
            per_mode_simple='Totals'
        ).get_data_frames()[0]
        
        cols = list(df.columns)
        print(f"Columns ({len(cols)}): {cols}")
        
        matches = [c for c in cols if "AND" in c.upper() or "PF" in c.upper()]
        if matches:
            print(f"Potential Matches: {matches}")
        else:
            print("No fuzzy matches found.")
            
    except Exception as e:
        print(f"Error: {e}")

# Check likely candidates
check_endpoint("Misc")
check_endpoint("Scoring")
check_endpoint("Advanced")
check_endpoint("Usage")
check_drives()
