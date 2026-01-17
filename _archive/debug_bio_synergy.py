import sys
import io
from nba_api.stats.endpoints import leaguedashplayerbiostats, synergyplaytypes
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("--- Inspecting BioStats Columns ---")
try:
    df = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
        season='2024-25',
        per_mode_simple='Totals',
        timeout=30
    ).get_data_frames()[0]
    print(list(df.columns))
except Exception as e:
    print(f"Error Bio: {e}")

print("\n--- Inspecting Synergy (Transition) Columns ---")
try:
    df = synergyplaytypes.SynergyPlayTypes(
        season='2024-25',
        play_type_nullable='Transition',
        type_grouping_nullable='offensive',
        per_mode_simple='Totals',
        player_or_team_abbreviation='P',
        timeout=30
    ).get_data_frames()[0]
    print(list(df.columns))
except Exception as e:
    print(f"Error Synergy: {e}")
