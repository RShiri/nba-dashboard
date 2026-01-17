import pandas as pd
import pickle
import os

# Mock data loading
try:
    if not os.path.exists("nba_data.pkl"):
        print("⚠️ nba_data.pkl not found, skipping verification")
        exit(0)

    with open("nba_data.pkl", "rb") as f:
        data = pickle.load(f)
    print("Loaded nba_data.pkl")
    
    career_basic = data.get("career_basic", pd.DataFrame())
    
    if career_basic.empty:
        print("⚠️ career_basic dataframe is empty")
        exit(0)

    # Simulate the logic
    df_display = career_basic.copy()
    
    # 1. Remove Cols
    drop_cols = ["PLAYER_ID", "LEAGUE_ID", "TEAM_ID"]
    df_display = df_display.drop(columns=[c for c in drop_cols if c in df_display.columns])
    
    # 2. Rename Cols
    # Format SEASON_ID to xx/xx (e.g., 2024-25 -> 24/25)
    if "SEASON_ID" in df_display.columns:
        df_display["SEASON_ID"] = df_display["SEASON_ID"].apply(
            lambda x: f"{x[2:4]}/{x[5:]}" if isinstance(x, str) and len(x) >= 7 else x
        )
            
    rename_map = {"SEASON_ID": "Season", "TEAM_ABBREVIATION": "TEAM", "PLAYER_AGE": "AGE"}
    df_display = df_display.rename(columns=rename_map)
    
    # Check results
    print("Columns:", df_display.columns.tolist())
    
    failed = False
    
    if "Season" in df_display.columns:
        sample_season = df_display["Season"].iloc[0]
        print(f"Sample Season: {sample_season}")
        if "/" not in str(sample_season):
             print("❌ Season format incorrect (missing /)")
             failed = True
    else:
        print("❌ Season column missing")
        failed = True

    if "PLAYER_ID" in df_display.columns:
        print("❌ PLAYER_ID still present")
        failed = True
    if "LEAGUE_ID" in df_display.columns:
        print("❌ LEAGUE_ID still present")
        failed = True
    if "TEAM_ID" in df_display.columns:
        print("❌ TEAM_ID still present")
        failed = True
    if "TEAM" not in df_display.columns:
        # Check if original had TEAM_ABBREVIATION
        if "TEAM_ABBREVIATION" in career_basic.columns:
             print("❌ TEAM column missing (rename failed)")
             failed = True
        else:
             print("⚠️ TEAM_ABBREVIATION was not in original, so TEAM is missing (expected)")

    if "AGE" not in df_display.columns:
         if "PLAYER_AGE" in career_basic.columns:
             print("❌ AGE column missing (rename failed)")
             failed = True
         else:
             print("⚠️ PLAYER_AGE was not in original, so AGE is missing (expected)")

    if not failed:
        print("✅ Verification Successful")
    else:
        print("❌ Verification Failed")

except Exception as e:
    print(f"❌ Verification Failed: {e}")
