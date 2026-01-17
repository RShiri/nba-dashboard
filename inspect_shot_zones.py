import pandas as pd
import pickle
import os

try:
    with open("nba_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    shot_charts = data.get("shot_charts", {})
    if shot_charts:
        # Get first available season
        k = list(shot_charts.keys())[0]
        df = shot_charts[k]
        print(f"Columns in shot data ({k}):")
        print(df.columns.tolist())
        
        print("\nUnique Values in SHOT_ZONE_BASIC:")
        print(df["SHOT_ZONE_BASIC"].unique())
        print("\nUnique Values in SHOT_ZONE_AREA:")
        print(df["SHOT_ZONE_AREA"].unique())
        print("\nUnique Values in SHOT_ZONE_RANGE:")
        print(df["SHOT_ZONE_RANGE"].unique())
        
        # Check combined uniqueness
        print("\nCombined unique zones:")
        combined = df.groupby(["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA"]).size()
        print(combined)
    else:
        print("No shot charts found.")
        
except Exception as e:
    print(f"Error: {e}")
