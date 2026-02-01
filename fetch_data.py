"""
Offline Data Fetching Script for Deni Avdija Analytics
Refactored for Smart Incremental Loading & Data Integrity

Run this script manually to fetch all NBA data and save it to nba_data.pkl.
This only needs to be run when you want to update the data.

Usage:
    python fetch_data.py
"""

import pandas as pd
from nba_api.stats.endpoints import (
    PlayerCareerStats,
    PlayerDashboardByYearOverYear,
    PlayerGameLog,
    shotchartdetail,
    leaguedashplayerstats,
    scoreboardv2,
)
from nba_api.stats.static import players
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import time
from requests.exceptions import RequestException
import sys

# Constants
PLAYER_NAME = "Deni Avdija"
PLAYER_ID = 1630166  # Known ID for Deni Avdija
SEASONS_SHOT_MAPS = ["2022-23", "2023-24", "2024-25", "2025-26"]
ALL_STAR_NAMES = [
    # User's curated elite comparison list (2025-26)
    "Giannis Antetokounmpo",
    "Jaylen Brown",
    "Jalen Brunson",
    "Cade Cunningham",
    "Tyrese Maxey",
    "Stephen Curry",
    "Luka Doncic",
    "Shai Gilgeous-Alexander",
    "Nikola Jokic",
    "Victor Wembanyama",
]

OUTPUT_FILE = "nba_data.pkl"


def is_valid_df(df) -> bool:
    """Strict validation: ensure DataFrame is not None and NOT empty."""
    return df is not None and not df.empty


def get_player_id(full_name: str = PLAYER_NAME) -> int:
    """Resolve NBA player ID."""
    try:
        hits = players.find_players_by_full_name(full_name)
        if hits:
            return hits[0]["id"]
    except Exception as e:
        print(f"⚠️  Could not resolve player ID from name, using known ID: {e}")
    return PLAYER_ID


def local_patch_career_stats(career_df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Locally update 2025-26 career stats from game logs to avoid re-fetching."""
    if logs.empty or career_df.empty:
        return career_df
    
    # Ensure logs columns are numeric for aggregation
    cols_to_agg = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PF"]
    # We assume fetch_game_logs already converted them, but safety check:
    for c in cols_to_agg:
        if c in logs.columns:
            logs[c] = pd.to_numeric(logs[c], errors="coerce").fillna(0)

    true_gp = len(logs)
    if true_gp == 0: return career_df

    # Calculate Fresh Averages
    updated_row = {
        "GP": true_gp,
        "GS": true_gp, # Assuming all starts for simplicity or need real GS logic? Logs usually don't have GS easily unless inferred. Keeping explicit for now.
        "MIN": logs["MIN"].mean() if "MIN" in logs.columns else 0,
        "PTS": logs["PTS"].mean(),
        "REB": logs["REB"].mean(),
        "AST": logs["AST"].mean(),
        "STL": logs["STL"].mean() if "STL" in logs.columns else 0,
        "BLK": logs["BLK"].mean() if "BLK" in logs.columns else 0,
        "TOV": logs["TOV"].mean() if "TOV" in logs.columns else 0,
        "PF": logs["PF"].mean() if "PF" in logs.columns else 0,
        "FGM": logs["FGM"].mean() if "FGM" in logs.columns else 0,
        "FGA": logs["FGA"].mean() if "FGA" in logs.columns else 0,
        "FG3M": logs["FG3M"].mean() if "FG3M" in logs.columns else 0,
        "FG3A": logs["FG3A"].mean() if "FG3A" in logs.columns else 0,
        "FTM": logs["FTM"].mean() if "FTM" in logs.columns else 0,
        "FTA": logs["FTA"].mean() if "FTA" in logs.columns else 0,
        # Recalculate PCTs from Totals (Sum)
        "FG_PCT": (logs["FGM"].sum() / logs["FGA"].sum()) if logs["FGA"].sum() > 0 else 0,
        "FG3_PCT": (logs["FG3M"].sum() / logs["FG3A"].sum()) if logs["FG3A"].sum() > 0 else 0,
        "FT_PCT": (logs["FTM"].sum() / logs["FTA"].sum()) if logs["FTA"].sum() > 0 else 0,
    }

    mask = career_df["SEASON_ID"] == "2025-26"
    if mask.any():
        idx = career_df.index[mask][0]
        # Update columns
        for col, val in updated_row.items():
            if col in career_df.columns:
                career_df.at[idx, col] = val
                
    return career_df


def fetch_career_basic(player_id: int) -> pd.DataFrame:
    """Fetch career basic per-season stats."""
    print("📊 Fetching Deni's Career Basic Stats...")
    try:
        df = PlayerCareerStats(player_id=player_id, per_mode36="PerGame").get_data_frames()[0]
        df = df.copy()
        df["SEASON_ID"] = df["SEASON_ID"].astype(str)
        print(f"✅ Career Basic Stats: {len(df)} seasons")
        return df
    except Exception as e:
        print(f"❌ Error fetching career basic stats: {e}")
        return pd.DataFrame()


def fetch_career_advanced(player_id: int) -> pd.DataFrame:
    """Fetch career advanced per-season stats."""
    print("📊 Fetching Deni's Career Advanced Stats...")
    try:
        adv = PlayerDashboardByYearOverYear(
            player_id=player_id,
            per_mode_detailed="PerGame",
            measure_type_detailed="Advanced",
        ).get_data_frames()[1]
        adv = adv.copy()
        adv["SEASON_ID"] = adv["GROUP_VALUE"].astype(str)
        print(f"✅ Career Advanced Stats: {len(adv)} seasons")
        return adv
    except Exception as e:
        print(f"❌ Error fetching career advanced stats: {e}")
        return pd.DataFrame()


def fetch_game_logs(player_id: int, season: str) -> pd.DataFrame:
    """Fetch game logs for a target season."""
    print(f"📅 Fetching Game Logs for {season}...")
    try:
        logs = PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        logs = logs.copy()
        
        if "SEASON_ID" in logs.columns:
            logs["SEASON_ID"] = logs["SEASON_ID"].astype(str)
        
        if "GAME_DATE" in logs.columns:
            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
        
        for col in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PF"):
            if col in logs.columns:
                logs[col] = pd.to_numeric(logs[col], errors="coerce")
        
        print(f"✅ {season} Game Logs: {len(logs)} games")
        return logs
    except Exception as e:
        print(f"❌ Error fetching {season} game logs: {e}")
        return pd.DataFrame()


def fetch_shot_data(player_id: int, season: str) -> pd.DataFrame:
    """Fetch shot chart data for a player and season."""
    print(f"  🏀 Fetching Shot Chart for {season}...")
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            context_measure_simple="FGA",
            season_nullable=season,
        ).get_data_frames()[0]
        shot_df = shot_data.copy()
        print(f"    ✅ {season}: {len(shot_df)} shots")
        return shot_df
    except Exception as e:
        print(f"    ❌ Error fetching shot data for {season}: {e}")
        return pd.DataFrame()


def fetch_allstar_stats(season="2024-25") -> pd.DataFrame:
    """Fetch stats for All-Stars using league API (FAST). Supports dynamic season."""
    print(f"⭐ Fetching All-Star Stats ({season})...")
    try:
        # Single API call to get ALL players for the season
        league_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        
        # Filter to only All-Stars by matching player names
        allstar_df = league_stats[
            league_stats["PLAYER_NAME"].isin(ALL_STAR_NAMES)
        ].copy()
        
        if allstar_df.empty:
            print("⚠️  No All-Stars found in league data")
            return pd.DataFrame()
            
        # Select and rename columns, round to 1 decimal
        cols_to_select = ["PLAYER_NAME", "GP", "PTS", "REB", "AST"]
        
        # Check for various casings of STL, BLK, TOV
        for stat, targets in [("STL", ["STL", "stl", "STEALS"]), ("BLK", ["BLK", "blk", "BLOCKS"]), ("TOV", ["TOV", "tov", "TURNOVERS"])]:
            for t in targets:
                if t in allstar_df.columns:
                    cols_to_select.append(t)
                    break
                    
        result_df = allstar_df[[col for col in cols_to_select if col in allstar_df.columns]].copy()
        
        # Normalize column names
        rename_map = {c: c.upper() if c not in ["STL", "BLK", "TOV"] else c for c in result_df.columns}
        # Force standard short names if long ones were found
        for c in result_df.columns:
            if c in ["STEALS", "stl"]: rename_map[c] = "STL"
            if c in ["BLOCKS", "blk"]: rename_map[c] = "BLK"
            if c in ["TURNOVERS", "tov"]: rename_map[c] = "TOV"
            
        result_df = result_df.rename(columns=rename_map)
        
        # Numeric cleanup
        for col in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(0).round(1)
                
        result_df["GP"] = result_df["GP"].astype(int)
        
        print(f"✅ All-Star Stats: {len(result_df)} players")
        return result_df
    except Exception as e:
        print(f"❌ Error fetching All-Star stats: {e}")
        return pd.DataFrame()


def fetch_league_ft_stats(season="2025-26") -> pd.DataFrame:
    """Fetch league-wide stats to determine FT leaders."""
    print(f"📊 Fetching League FT Stats for {season}...")
    try:
        # Fetch ALL players
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="Totals"
        ).get_data_frames()[0]
        
        # Keep relevant columns for FT leaderboard
        cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "FTM", "FTA", "FT_PCT"]
        df = df[[c for c in cols if c in df.columns]].copy()
        
        # Sort by FTM descending
        df = df.sort_values("FTM", ascending=False)
        
        print(f"✅ League Stats: {len(df)} players fetched.")
        return df
    except Exception as e:
        print(f"❌ Error fetching League FT stats: {e}")
        return pd.DataFrame()


def fetch_drives_data(season="2025-26") -> pd.DataFrame:
    """Fetch Drives data for Rim Pressure table."""
    print(f"🚗 Fetching Drives Data ({season})...")
    try:
        from nba_api.stats.endpoints import leaguedashptstats
        df = leaguedashptstats.LeagueDashPtStats(
            season=season,
            pt_measure_type='Drives',
            player_or_team='Player',
            per_mode_simple='PerGame'
        ).get_data_frames()[0]
        
        cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "DRIVES", "DRIVE_PTS"]
        df = df[[c for c in cols if c in df.columns]].copy()
        print(f"✅ Drives Data: {len(df)} players")
        return df
    except Exception as e:
        print(f"❌ Error fetching drives data: {e}")
        return pd.DataFrame()


def fetch_misc_stats(season="2025-26") -> pd.DataFrame:
    """Fetch Misc Data (PFD + AND_ONES)."""
    print(f"🧩 Fetching Misc Stats (PFD + And-Ones) for {season}...")
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        
        # 'Misc' endpoint 
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense='Misc',
            per_mode_detailed='Totals',
            timeout=30
        ).get_data_frames()[0]
        
        target_cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "PFD", "AND_ONES"]
        found_cols = [c for c in target_cols if c in df.columns]
        
        if not found_cols:
             print("   ⚠️  No relevant columns found in Misc stats.")
             return pd.DataFrame()
             
        df = df[found_cols].copy()
        
        # Debug print
        print(f"   ℹ️  Columns found: {found_cols}")
        
        # Sort by PFD by default if exists
        if "PFD" in df.columns:
            df = df.sort_values("PFD", ascending=False)
            
        print(f"✅ Misc Data: {len(df)} players")
        return df
    except Exception as e:
        print(f"❌ Error fetching Misc data: {e}")
        return pd.DataFrame()


def fetch_passing_data(season="2025-26") -> pd.DataFrame:
    """Fetch Passing data and calculate AST_3P."""
    print(f"🏀 Fetching Passing Data ({season})...")
    try:
        from nba_api.stats.endpoints import leaguedashptstats
        df = leaguedashptstats.LeagueDashPtStats(
            season=season,
            pt_measure_type='Passing',
            player_or_team='Player',
            per_mode_simple='Totals' # Totals for "Sniper Finders" usually makes sense, but user asked for "Assists to 3P".
            # Let's use Totals to find the volume finders.
        ).get_data_frames()[0]
        
        cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "AST", "AST_PTS_CREATED"]
        df = df[[c for c in cols if c in df.columns]].copy()
        
        # CALCULATION: AST_3P
        # AST_PTS_CREATED - (2 * AST)
        df["AST_3P"] = df["AST_PTS_CREATED"] - (2 * df["AST"])
        
        print(f"✅ Passing Data: {len(df)} players")
        return df
    except Exception as e:
        print(f"❌ Error fetching passing data: {e}")
        return pd.DataFrame()


def fetch_team_star_dependence(season="2025-26") -> pd.DataFrame:
    """
    Calculate Team vs. Star Dependence (Heliocentric).
    Returns DataFrame with [TEAM_NAME, TEAM_PPG, STAR_NAME, STAR_OUTPUT]
    """
    print(f"☀️ Fetching Heliocentric Data ({season})...")
    try:
        from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats
        
        # 1. Team Stats (For PPG)
        team_df = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed='PerGame'
        ).get_data_frames()[0]
        
        # 2. Player Stats (PTS, AST)
        player_df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed='PerGame'
        ).get_data_frames()[0]
        
        # Process Teams
        teams_map = {} # ID -> {Name, PPG}
        for _, row in team_df.iterrows():
            teams_map[row['TEAM_ID']] = {
                'TEAM_NAME': row['TEAM_NAME'],
                'TEAM_PPG': row['PTS']
            }
            
        # Process Players & Find Max per Team
        # We need to group by TEAM_ID
        
        # Calc Output
        # output = PTS + (AST * 2.3)
        player_df["TOTAL_OUTPUT"] = player_df["PTS"] + (player_df["AST"] * 2.3)
        
        # Group by Team ID and find max
        # We also need to keep Player Name
        results = []
        
        # Iterate over unique teams in player_df
        for tid in player_df['TEAM_ID'].unique():
            if tid not in teams_map: continue
            
            team_roster = player_df[player_df['TEAM_ID'] == tid]
            if team_roster.empty: continue
            
            # Find star
            star = team_roster.loc[team_roster['TOTAL_OUTPUT'].idxmax()]
            
            t_info = teams_map[tid]
            results.append({
                "TEAM_NAME": t_info['TEAM_NAME'],
                "TEAM_PPG": t_info['TEAM_PPG'],
                "STAR_NAME": star['PLAYER_NAME'],
                "STAR_OUTPUT": star['TOTAL_OUTPUT']
            })
            
        final_df = pd.DataFrame(results)
        final_df = final_df.sort_values("TEAM_PPG", ascending=False)
        
        print(f"✅ Heliocentric Data: {len(final_df)} teams")
        return final_df

    except Exception as e:
        print(f"❌ Error fetching heliocentric data: {e}")
        return pd.DataFrame()


def fetch_allstar_detailed_stats(season="2024-25", existing_df=None) -> pd.DataFrame:
    """Fetch detailed stats for All-Stars including USG_PCT and TS_PCT."""
    print(f"⭐ Fetching All-Star Detailed Stats ({season})...")
    total_players = len(ALL_STAR_NAMES)
    print(f"   ⚠️  Checking detailed stats for {total_players} players...")
    
    all_stats = []
    
    # Pre-load existing data for lookup
    existing_map = {}
    if is_valid_df(existing_df):
        # Convert to dict for faster lookups {PlayerName: {Stats}}
        # Ensure we only keep data that actually matches the season if possible? 
        # But existing_df comes from a key specific to that season, so we trust it.
        for _, row in existing_df.iterrows():
            existing_map[row["PLAYER_NAME"]] = row.to_dict()

    for idx, player_name in enumerate(ALL_STAR_NAMES, 1):
        # OPTIMIZATION: Check if we already have this player
        if player_name in existing_map:
            all_stats.append(existing_map[player_name])
            print("♻️", end="", flush=True)
            continue
            
        # If not, fetch:
        try:
            hits = players.find_players_by_full_name(player_name)
            if hits:
                pid = hits[0]["id"]
                # Fetch Advanced Stats
                adv = PlayerDashboardByYearOverYear(
                    player_id=pid,
                    per_mode_detailed="PerGame",
                    measure_type_detailed="Advanced",
                ).get_data_frames()[1]
                
                # Fetch Basic Stats (backup if needed, but usually we use the summary df for basic)
                # Here we just want the advanced metrics for the current season
                
                # Filter for Target Season
                mask = adv["GROUP_VALUE"].astype(str).str.contains(season)
                season_data = adv[mask]
                
                if not season_data.empty:
                    row = season_data.iloc[0]
                    all_stats.append({
                        "PLAYER_NAME": player_name,
                        "USG_PCT": round(row.get("USG_PCT", 0), 3),
                        "TS_PCT": round(row.get("TS_PCT", 0), 3)
                    })
                    print(".", end="", flush=True)
                else:
                    print("x", end="", flush=True)
            
            # Rate limiting
            time.sleep(0.6)
            
        except Exception:
            print("!", end="", flush=True)
            
    print("\n✅ Detailed fetch complete.")
    return pd.DataFrame(all_stats)


def check_new_games(existing_logs: pd.DataFrame, team_id: int = 1610612757) -> bool:
    """
    Checks if there is a recently COMPLETED game that is missing from our logs.
    Scans from the last known game date up to today.
    """
    print("\n🕵️ Checking for recent completed games to sync...")
    
    # 1. Determine Start Date
    start_date = datetime(2025, 10, 22) # Start of 25-26 Season approx
    existing_game_ids = set()

    if is_valid_df(existing_logs):
        if "GAME_DATE" in existing_logs.columns:
            try:
                last_date = pd.to_datetime(existing_logs["GAME_DATE"]).max()
                if not pd.isna(last_date):
                    start_date = last_date + timedelta(days=1)
            except Exception:
                pass
        
        if "GAME_ID" in existing_logs.columns:
            # Create a set for O(1) lookups. Ensure strings.
            existing_game_ids = set(existing_logs["GAME_ID"].astype(str))
            
    # Cap search to last 14 days to avoid huge API loops if something weird happens
    limit_date = datetime.now() - timedelta(days=14)
    if start_date < limit_date:
        print(f"   ⚠️  Last game seems very old ({start_date.date()}). Limiting check to last 14 days.")
        start_date = limit_date

    end_date = datetime.now()
    
    # If up to date
    if start_date > end_date:
        print("   ✅ Logs are up to date (Next possible game is in future).")
        return False
        
    print(f"   📅 Scanning from {start_date.date()} to {end_date.date()}...")

    # 2. Iterate
    current_date = start_date
    while current_date <= end_date:
        d_str = current_date.strftime("%Y-%m-%d")
        try:
            # We use a broader Exception block because sometimes API times out
            board = scoreboardv2.ScoreboardV2(game_date=d_str).game_header.get_data_frame()
            
            if is_valid_df(board):
                # Filter for team
                game = board[(board['HOME_TEAM_ID'] == team_id) | (board['VISITOR_TEAM_ID'] == team_id)]
                
                if not game.empty:
                    status_text = game.iloc[0]['GAME_STATUS_TEXT'] # e.g. "Final"
                    game_id = str(game.iloc[0]['GAME_ID'])
                    print(f"   - {d_str}: Found game. Status='{status_text}' (ID: {game_id})")
                    
                    if "Final" in status_text:
                        # Check if we have it
                        if game_id not in existing_game_ids:
                            print(f"   🚨 NEW FINAL GAME FOUND! (ID: {game_id}). Triggering update.")
                            return True
                        else:
                            print(f"   ✅ Game {game_id} already in logs.")
        except Exception as e:
            print(f"   ⚠️ Error checking {d_str}: {e}")
            
        current_date += timedelta(days=1)
        # Small sleep to be nice to API
        time.sleep(0.3)
    
    print("   ✨ No new missing completed games found.")
    return False


def smart_update(force_refresh=False):
    """
    Main update logic. 
    Returns a status string message.
    """
    # Force UTF-8 for Windows
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    print("=" * 60)
    print("🚀 Starting NBA Data Smart Update")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load Existing Data
    existing_data = {}
    if Path(OUTPUT_FILE).exists():
        try:
            print(f"📂 Loading existing data from {OUTPUT_FILE}...")
            with open(OUTPUT_FILE, "rb") as f:
                existing_data = pickle.load(f)
            print(f"✅ Loaded {len(existing_data)} keys.")
        except Exception as e:
            print(f"⚠️  Corrupt data file, starting fresh: {e}")
            existing_data = {}

    player_id = get_player_id()

    # 1b. Load existing 25-26 Logs for Comparison
    logs_25_26 = existing_data.get("game_logs_2025_26", pd.DataFrame())
    
    # 2. DECIDE IF WE NEED TO UPDATE
    should_update = force_refresh
    if not should_update:
        # Check if we are missing a recent game
        should_update = check_new_games(logs_25_26)

    if should_update:
        # 2a. Fetch Current Season Game Logs
        print("\n📅 Fetching 2025-26 Game Logs (New/Forced)...")
        new_logs = fetch_game_logs(player_id, "2025-26")
        if is_valid_df(new_logs):
            logs_25_26 = new_logs
        else:
             print("   ⚠️  Fetch failed? Keeping old logs.")
    else:
        print("\n✅ Data is up-to-date (No new games found). Skipping major fetch.")
    
    # Check actual number of games played
    actual_games_played = len(logs_25_26)
    
    # 3. Smart Career Fetch
    # Check if we can skip career fetch
    need_career_fetch = True
    
    # Use helper is_valid_df to ensure cache is valid
    if not should_update and "career_basic" in existing_data and is_valid_df(existing_data["career_basic"]):
        cached_df = existing_data["career_basic"]
        # Find 2025-26 row
        current_season_row = cached_df[cached_df["SEASON_ID"] == "2025-26"]
        
        if not current_season_row.empty:
            cached_gp = int(current_season_row.iloc[0]["GP"])
            print(f"   🔍 Integrity Check: Cache says {cached_gp} GP, Logs say {actual_games_played} GP")
            
            if cached_gp >= actual_games_played:
                print("   ♻️  Career stats match game logs. Skipping fetch.")
                career_basic = cached_df
                career_adv = existing_data.get("career_advanced", pd.DataFrame())
                need_career_fetch = False
            else:
                print("   ⚠️  Career stats lagging. Applying local patch from logs...")
                career_basic = local_patch_career_stats(cached_df, logs_25_26)
                career_adv = existing_data.get("career_advanced", pd.DataFrame())
                need_career_fetch = False
        else:
             print("   ⚠️  No 2025-26 entry in cached career stats. Forcing refresh.")
    else:
        print("   ⚠️  Found empty or invalid career_basic in cache (or forced). Forcing refresh.")
    
    if need_career_fetch:
        career_basic = fetch_career_basic(player_id)
        career_adv = fetch_career_advanced(player_id)
        
    # 3b. Fetch previous season logs if missing
    if not should_update and "game_logs_2024_25" in existing_data and is_valid_df(existing_data["game_logs_2024_25"]):
        logs_24_25 = existing_data["game_logs_2024_25"]
    else:
        print("   ⚠️  Cached 2024-25 logs invalid/empty. Refreshing...")
        logs_24_25 = fetch_game_logs(player_id, "2024-25")

    # 3c. Fetch 2023-24 season logs (requested for history)
    if not should_update and "game_logs_2023_24" in existing_data and is_valid_df(existing_data["game_logs_2023_24"]):
        logs_23_24 = existing_data["game_logs_2023_24"]
    else:
        print("   ⚠️  Fetch 2023-24 logs (New history)...")
        logs_23_24 = fetch_game_logs(player_id, "2023-24")

    # 4. Shot Charts
    print("\n🏀 Fetching Shot Charts...")
    shot_charts = existing_data.get("shot_charts", {})
    if shot_charts is None: shot_charts = {}
    
    # Always fetch current season if updating
    if should_update:
        print("   🔄 Fetching 2025-26 Shot Chart (New/Forced)...")
        shot_charts["2025-26"] = fetch_shot_data(player_id, "2025-26")
    elif "2025-26" not in shot_charts:
         print("   ⚠️  Missing 2025-26 shot chart? Fetching...")
         shot_charts["2025-26"] = fetch_shot_data(player_id, "2025-26")
    
    # Smart fetch past seasons
    for season in ["2022-23", "2023-24", "2024-25"]:
        # Strict check: exists in dict AND is valid
        if not should_update and season in shot_charts and is_valid_df(shot_charts[season]):
            print(f"   ♻️  Using cached {season} chart")
        else:
            print(f"   ⚠️  Empty/Invalid cached data for {season} chart. Re-fetching...")
            shot_charts[season] = fetch_shot_data(player_id, season)

    # 5. All-Star Data (Static - 2024-25)
    print("\n⭐ Checking All-Star Data...")
    should_fetch_allstar = True
    
    # Safety: Default to existing if available
    allstar_stats = existing_data.get("allstar_stats", pd.DataFrame())
    allstar_detailed = existing_data.get("allstar_detailed_stats", pd.DataFrame())

    # OLD All-Star (2024-25 Benchmark) - Frozen
    if is_valid_df(allstar_stats):
        print(f"   ❄️  All-Star Benchmark (24-25) frozen/cached. Skipping.")
    else:
        print("   ⚠️  No All-Star Benchmark (24-25) found. Fetching (One-time)...")
        new_bench = fetch_allstar_stats("2024-25")
        if is_valid_df(new_bench):
            allstar_stats = new_bench
            # PASS EXISTING for Benchmark (though usually we won't even be here if it's frozen)
            # But let's say we had basic stats but missing detailed?
            allstar_detailed = fetch_allstar_detailed_stats("2024-25", existing_df=allstar_detailed)
    
    # 5b. All-Star RACE (2025-26) - Live
    print("\n🏎️  Checking All-Star Race (2025-26)...")
    allstar_stats_26 = existing_data.get("allstar_stats_26", pd.DataFrame())
    allstar_detailed_26 = existing_data.get("allstar_detailed_26", pd.DataFrame())

    if should_update:
        print("   🔄 Updating All-Star Race data (2025-26)...")
        # Fetch stats for the SAME cohort but for current season
        new_race = fetch_allstar_stats("2025-26")
        if is_valid_df(new_race):
             allstar_stats_26 = new_race
             # Only fetch detailed if basic succeeded
             # PASS EXISTING 25-26 Data to skip re-fetching known players
             allstar_detailed_26 = fetch_allstar_detailed_stats("2025-26", existing_df=allstar_detailed_26)
        else:
             print("   ⚠️  Race fetch failed (maybe season too early?). Keeping old data.")
    else:
        print("   ✅ Race data up-to-date.")

    # 6. League FT Stats (2025-26)
    print("\n📊 Checking League FT Stats...")
    # We want to refresh this if we are doing a fresh fetch run, or if it's missing
    league_ft = existing_data.get("league_ft_stats", pd.DataFrame())

    # We want to force refresh it if we are doing a general update as it changes often
    # But let's assume if we are running this script, we want the latest FT leaderboard.
    if should_update:
        new_league_ft = fetch_league_ft_stats("2025-26")
        if is_valid_df(new_league_ft):
            league_ft = new_league_ft
        elif is_valid_df(league_ft):
             print("   ⚠️  League FT fetch failed, using old cache.")
        else:
             print("   ❌ League FT fetch failed and no cache.")

    # 7. League Trends Data (New)
    print("\n📈 Fetching League Trends Data...")
    drives_df = existing_data.get("drives_data", pd.DataFrame())
    misc_df = existing_data.get("misc_stats", pd.DataFrame())
    passing_df = existing_data.get("passing_data", pd.DataFrame())
    heliocentric_df = existing_data.get("heliocentric_data", pd.DataFrame())
    
    if should_update:
        drives_df = fetch_drives_data("2025-26")
        misc_df = fetch_misc_stats("2025-26")
        passing_df = fetch_passing_data("2025-26")
        heliocentric_df = fetch_team_star_dependence("2025-26")
    else:
        # Granular checks - only fetch what is missing
        if drives_df.empty:
            print("   ⚠️  Missing Drives data, fetching...")
            drives_df = fetch_drives_data("2025-26")
        else:
             print("   ♻️  Drives data cached.")

        if misc_df.empty:
            print("   ⚠️  Missing Misc data, fetching...")
            misc_df = fetch_misc_stats("2025-26")
        else:
             print("   ♻️  Misc data cached.")

        if passing_df.empty:
            print("   ⚠️  Missing Passing data, fetching...")
            passing_df = fetch_passing_data("2025-26")
        else:
             print("   ♻️  Passing data cached.")

        if heliocentric_df.empty:
            print("   ⚠️  Missing Heliocentric data, fetching...")
            heliocentric_df = fetch_team_star_dependence("2025-26")
        else:
             print("   ♻️  Heliocentric data cached.")

    # 8. Save
    data_dict = {
        "career_basic": career_basic,
        "career_advanced": career_adv,
        "game_logs_2025_26": logs_25_26,
        "game_logs_2024_25": logs_24_25,
        "game_logs_2023_24": logs_23_24,
        "shot_charts": shot_charts,
        "allstar_stats": allstar_stats,
        "allstar_stats": allstar_stats,
        "allstar_detailed_stats": allstar_detailed,
        "allstar_stats_26": allstar_stats_26,
        "allstar_detailed_26": allstar_detailed_26,
        "league_ft_stats": league_ft,
        # New Keys
        "drives_data": drives_df,
        "misc_stats": misc_df,
        "passing_data": passing_df,
        "heliocentric_data": heliocentric_df,
        
        "fetched_at": datetime.now().isoformat(),
        "player_id": player_id,
        "player_name": PLAYER_NAME,
    }

    try:
        print(f"\n💾 Saving to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"✅ Data saved successfully! (Keys: {list(data_dict.keys())})")
        
        # Auto-commit and push to Git if successful
        try:
            print("\n🔄 Attempting Git auto-update...")
            import subprocess
            from pathlib import Path
            
            script_dir = Path(__file__).parent
            
            # Run the auto_update script
            result = subprocess.run(
                [sys.executable, "auto_update.py"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅ Git auto-update successful!")
                print(result.stdout)
            else:
                print("⚠️  Git auto-update skipped or failed (this is OK if not using Git)")
                if result.stderr:
                    print(f"   Details: {result.stderr[:200]}")
                    
        except Exception as git_error:
            print(f"⚠️  Git auto-update failed: {git_error}")
            print("   (Data was saved successfully, only Git sync failed)")
            
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return f"Error: {e}"

    print("\n" + "="*60)
    print("✅ FETCH COMPLETE. Data updated.")
    print("="*60)
    return "Data updated successfully"

if __name__ == "__main__":
    smart_update()
