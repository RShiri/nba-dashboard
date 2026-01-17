import pickle
import pandas as pd

try:
    with open('nba_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    allstar_df = data.get('allstar_stats', pd.DataFrame())
    if not allstar_df.empty:
        print('All-Star Stats Columns:')
        print(list(allstar_df.columns))
        print('\nFirst few rows:')
        print(allstar_df.head(3))
    else:
        print('No All-Star data found or DataFrame is empty')
except Exception as e:
    print(f'Error: {e}')

