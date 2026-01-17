import pickle
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    with open("nba_data.pkl", "rb") as f:
        data = pickle.load(f)
        
    df = data.get("and_one_data", pd.DataFrame())
    print("\n--- Verifying 'and_one_data' (Now PFD) ---")
    if df.empty:
        print("❌ DataFrame is empty!")
    else:
        print(f"✅ Row Count: {len(df)}")
        print(f"✅ Columns: {list(df.columns)}")
        if "PFD" in df.columns:
            print("✅ 'PFD' column found.")
            print(df.head())
        else:
            print("❌ 'PFD' column MISSING.")
            
except Exception as e:
    print(f"Error: {e}")
