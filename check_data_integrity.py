import pandas as pd
import pickle
import os

DATA_FILE = "nba_data.pkl"

if not os.path.exists(DATA_FILE):
    print(f"❌ {DATA_FILE} not found.")
    exit(1)

try:
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    
    print(f"Keys in data: {list(data.keys())}")
    
    logs_23 = data.get("game_logs_2023_24", pd.DataFrame())
    
    if logs_23 is None:
        print("❌ game_logs_2023_24 is None")
    elif logs_23.empty:
        print("❌ game_logs_2023_24 is Empty DataFrame")
    else:
        print(f"✅ game_logs_2023_24 found with {len(logs_23)} records.")
        print(logs_23.head(2))

except Exception as e:
    print(f"❌ Error loading data: {e}")
