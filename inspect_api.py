import sys
import io
import inspect
from nba_api.stats.endpoints import leaguedashplayerstats

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("--- Inspecting LeagueDashPlayerStats arguments ---")
sig = inspect.signature(leaguedashplayerstats.LeagueDashPlayerStats.__init__)
for name, param in sig.parameters.items():
    print(f"{name}: {param.default}")

print("\n--- Trying to force MeasureType='Misc' via kwargs check ---")
# Try passing 'measure_type_detailed_defense' as 'Misc' again but maybe with debug on? 
# No, let's try 'measure_type_detailed_defense'='Scoring' or something to see if it changes ANYTHING.
