import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def analyze_zone_angles():
    # Load data
    DATA_FILE = "nba_data.pkl"
    if not Path(DATA_FILE).exists():
        print("No data file found.")
        return

    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)

    # Combine all shot charts
    all_shots = []
    # Dictionary of season -> DataFrame
    shot_charts = data.get("shot_charts", {})
    if not shot_charts:
        print("No shot chart data found.")
        return

    for k, df in shot_charts.items():
        if not df.empty:
            all_shots.append(df)
    
    if not all_shots:
        print("No shot data available.")
        return
        
    df = pd.concat(all_shots)
    
    # Filter for standard zones
    # 'Mid-Range', 'Above the Break 3', 'Right Corner 3', 'Left Corner 3'
    # Exclude Restricted Area and Paint for angle analysis
    df = df[df["SHOT_ZONE_BASIC"].isin(["Mid-Range", "Above the Break 3", "Right Corner 3", "Left Corner 3"])]
    
    # Calculate Angle (degrees)
    # LOC_X, LOC_Y. Hoop at (0,0)?
    # Usually NBA API: X is left-right (-250 to 250), Y is baseline-top (-47 to 420).
    # Angle 0 is Right? 90 is Top?
    # arctan2(y, x) -> radians. 
    # But usually 0 degrees is along positive X axis.
    df["angle"] = np.degrees(np.arctan2(df["LOC_Y"], df["LOC_X"]))
    
    # Adjust negative angles to 0-360 or keep as -180 to 180?
    # Center is near 90.
    
    print(f"Analyzing {len(df)} shots...")
    
    zones = ["Right Side(R)", "Right Side Center(RC)", "Center(C)", "Left Side Center(LC)", "Left Side(L)"]
    
    for zone in zones:
        subset = df[df["SHOT_ZONE_AREA"] == zone]
        if subset.empty:
            print(f"{zone}: No shots")
            continue
            
        min_a = subset["angle"].min()
        max_a = subset["angle"].max()
        mean_a = subset["angle"].mean()
        
        print(f"ZONE: {zone:25} | Range: [{min_a:.1f}, {max_a:.1f}] deg | Mean: {mean_a:.1f}")

if __name__ == "__main__":
    analyze_zone_angles()
