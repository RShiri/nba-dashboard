import pickle
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    with open("nba_data.pkl", "rb") as f:
        data = pickle.load(f)
        
    df = data.get("misc_stats", pd.DataFrame())
    print("\n--- Verifying 'misc_stats' ---")
    if df.empty:
        print("❌ DataFrame is empty!")
    else:
        print(f"✅ Row Count: {len(df)}")
        print(f"✅ Columns: {list(df.columns)}")
        
        if "PFD" in df.columns:
            print("✅ 'PFD' column found.")
        else:
            print("❌ 'PFD' column MISSING.")
            
        if "AND_ONES" in df.columns:
             print("✅ 'AND_ONES' column found.")
        else:
             print("❌ 'AND_ONES' column MISSING (Expected).")

except Exception as e:
    print(f"Error: {e}")
