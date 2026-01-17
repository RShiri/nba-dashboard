import sys
import io
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_param(param_name, value):
    print(f"\n--- Testing {param_name}={value} ---")
    try:
        kwargs = {
            'season': '2024-25',
            'per_mode_detailed': 'Totals',
            param_name: value
        }
        df = leaguedashplayerstats.LeagueDashPlayerStats(**kwargs).get_data_frames()[0]
        
        # Check columns
        cols = list(df.columns)
        if "AND_ONES" in cols:
            print("SUCCESS! 'AND_ONES' found.")
            print(df[["PLAYER_NAME", "AND_ONES"]].head())
        else:
            print(f"FAILED. 'AND_ONES' NOT found.")
            print(f"Columns found (first 10): {cols[:10]}...")
            if 'MeasureType' in kwargs:
                 print(f"   (Used MeasureType: {kwargs['MeasureType']})")
    except Exception as e:
        print(f"Error: {e}")

print("Starting tests...")

# Test 1: Current implementation (measure_type_detailed_defense='Misc')
test_param('measure_type_detailed_defense', 'Misc')

# Test 2: Using 'measure_type_detailed' (mapped to MeasureType?)
test_param('measure_type_detailed', 'Misc')
