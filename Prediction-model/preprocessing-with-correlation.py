import pandas as pd
import numpy as np
import glob
import os
import re
import time
from typing import List, Dict

# ==============================================================================
#  Configuration
# ==============================================================================
CURRENCIES = ["CARD", "GARR", "HEST", "JUMP", "LOGN", "SIMP"]
BASE_PATH = '../market-data/'
OUTPUT_PATH = '../processed-data/'

# ==============================================================================
#  Feature Engineering & Data Transformation Functions
# ==============================================================================

def create_univariate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a comprehensive set of features for a single currency's time-series."""
    # Ensure index is the integer time step for all calculations
    if 'index' in df.columns:
        df = df.set_index('index', drop=False).rename(columns={'index': 'time_step'})
    
    # --- Part 1: Intraday & Wick Features ---
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['body_pct'] = df['body'] / (df['range'] + 1e-6)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # --- Part 2: Lag Features ---
    lags_to_create = [1, 2, 3, 5, 7, 10]
    columns_to_lag = ['mid', 'range', 'body_pct'] 
    for col in columns_to_lag:
        for lag in lags_to_create:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # --- Part 3: Trend Features ---
    windows = [6, 12, 25, 50, 100, 200]
    features_to_average = ['close', 'range', 'body_pct']
    for col in features_to_average:
        for w in windows:
            df[f'{col}_sma_{w}'] = df[col].rolling(window=w).mean()
            df[f'{col}_ema_{w}'] = df[col].ewm(span=w, adjust=False).mean()

    # --- Part 4: Trend-Relative & Volatility Features ---
    for w in windows:
        df[f'close_div_ema_{w}'] = df['close'] / (df[f'close_ema_{w}'] + 1e-6)
    
    vol_windows = [12, 25, 50]
    for w in vol_windows:
        df[f'range_std_{w}'] = df['range'].rolling(window=w).std()

    # --- Part 5: Momentum Acceleration (MACD) ---
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

def get_dynamic_target_timestamps(t_now: int, n_targets: int = 4, interval: int = 15) -> List[int]:
    """Calculates the nearest future n_targets that are multiples of the interval."""
    first_target = ((t_now // interval) + 1) * interval
    return [first_target + i * interval for i in range(n_targets)]

# ==============================================================================
#  Main Pipeline Functions
# ==============================================================================

def discover_and_sort_files(base_path: str, currencies: List[str]) -> Dict[str, List[str]]:
    """Finds all data files and sorts them chronologically for each currency."""
    print("Step 1: Discovering and sorting data files...")
    files_by_currency = {}
    for curr in currencies:
        all_files = glob.glob(f'{base_path}{curr}/*.json')
        all_files.sort()
        files_by_currency[curr] = all_files
        print(f"  - Found {len(all_files)} files for {curr}.")
    return files_by_currency

def create_universal_master_dataset(files_by_currency: Dict[str, List[str]]) -> pd.DataFrame:
    """
    The main pipeline function to load, feature-engineer, merge, and stack data
    from all currencies into a single, model-ready master DataFrame.
    """
    print("\nStep 2: Engineering features for each currency independently...")
    featured_dfs = {}
    for currency, sorted_files in files_by_currency.items():
        print(f"  - Processing all files for {currency}...")
        df_currency_history = pd.concat([pd.read_json(f, orient='index') for f in sorted_files])
        df_currency_history.sort_index(inplace=True)
        df_currency_history.drop_duplicates(inplace=True)
        
        df_featured = create_univariate_features(df_currency_history.copy())
        featured_dfs[currency] = df_featured

    print("\nStep 3: Creating universal DataFrame with cross-asset features...")
    print("  - Renaming columns for merging...")
    for currency, df in featured_dfs.items():
        # The index is already 'time_step', we don't rename it
        df.rename(columns={col: f"{col}_{currency}" for col in df.columns if col != 'time_step'}, inplace=True)

    print("  - Merging all currency dataframes on 'time_step'...")
    list_of_dfs_to_merge = [df.drop(columns=[f'time_step_{c}']) for c, df in featured_dfs.items()]
    # Use reduce for a more efficient merge on multiple dataframes
    from functools import reduce
    universal_df = reduce(lambda left, right: pd.merge(left, right, on='time_step', how='inner'), list_of_dfs_to_merge)
    
    print(f"  - Universal DataFrame created with shape: {universal_df.shape}")

    print("  - Creating relative strength features...")
    close_cols = [f'close_{c}' for c in CURRENCIES]
    universal_df['market_avg_close'] = universal_df[close_cols].mean(axis=1)
    for curr in CURRENCIES:
        universal_df[f'rel_strength_{curr}'] = universal_df[f'close_{curr}'] / (universal_df['market_avg_close'] + 1e-6)

    print("\nStep 4: Stacking the universal dataset for multi-horizon training...")
    target_price_maps = {curr: df.set_index('time_step')[f'close_{curr}'] for curr, df in featured_dfs.items()}
    all_stacked_rows = []

    for t_now_step, feature_row in universal_df.iterrows():
        for currency_to_predict in CURRENCIES:
            target_time_steps = get_dynamic_target_timestamps(t_now_step)
            for T_target_step in target_time_steps:
                target_price = target_price_maps[currency_to_predict].get(T_target_step)
                if target_price is None:
                    continue
                
                new_row = feature_row.to_dict()
                new_row['currency_to_predict'] = currency_to_predict
                new_row['forecast_horizon'] = T_target_step - t_now_step
                new_row['target'] = target_price
                all_stacked_rows.append(new_row)

    universal_master_dataset = pd.DataFrame(all_stacked_rows)
    initial_rows = len(universal_master_dataset)
    # Drop rows with NaNs, which are typically at the beginning of the history
    universal_master_dataset.dropna(inplace=True)
    final_rows = len(universal_master_dataset)

    print(f"  - Final dataset created with shape: {universal_master_dataset.shape}")
    print(f"  - Dropped {initial_rows - final_rows:,} rows with NaN values.")
    
    return universal_master_dataset

# ==============================================================================
#  Main Execution Block
# ==============================================================================

def main():
    """Main function to run the entire preprocessing pipeline."""
    
    # Run the full pipeline
    files = discover_and_sort_files(BASE_PATH, CURRENCIES)
    master_dataset = create_universal_master_dataset(files)

    # Save the final, single dataset
    print("\nStep 5: Saving universal master dataset to Parquet file...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    if not master_dataset.empty:
        file_path = f'{OUTPUT_PATH}UNIVERSAL_master_dataset.parquet'
        print(f"  - Saving universal dataset ({len(master_dataset):,} rows) to: {file_path}")
        master_dataset.to_parquet(file_path, index=False)
        print("\n✅ Preprocessing complete. Dataset is ready for training.")
    else:
        print("🔴 ERROR: The final master dataset is empty. No file was saved.")


if __name__ == "__main__":
    # This block allows the script to be run from the command line
    main()