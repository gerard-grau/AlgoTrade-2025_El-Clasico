import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#  Configuration
# ==============================================================================
# Define which currencies' results to plot
CURRENCIES_TO_PLOT = ["CARD", "GARR", "HEST", "JUMP", "LOGN", "SIMP"]

# Define paths (should match your training script)
PROCESSED_DATA_PATH = '../../data/processed/'
MODELS_OUTPUT_PATH = '../../models/'

# ==============================================================================
#  Plotting Functions
# ==============================================================================

def plot_forecast_vs_actual(y_test, predictions, currency):
    """
    Plots the model's forecast (mean/median and prediction interval) against
    the actual values over time. This gives a great overview of performance.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 8))

    # A subset of the data is easier to see
    subset_size = 500
    y_test_subset = y_test.iloc[-subset_size:]
    preds_lower_subset = predictions['lower'][-subset_size:]
    preds_upper_subset = predictions['upper'][-subset_size:]
    preds_median_subset = predictions['median'][-subset_size:]
    
    # Plot the prediction interval (the "uncertainty cone")
    ax.fill_between(y_test_subset.index, preds_lower_subset, preds_upper_subset, 
                    color='skyblue', alpha=0.4, label='90% Prediction Interval')
    
    # Plot the actual values
    ax.plot(y_test_subset.index, y_test_subset, 
            color='navy', lw=2, label='Actual Price')

    # Plot the median forecast
    ax.plot(y_test_subset.index, preds_median_subset, 
            color='darkorange', linestyle='--', lw=2, label='Median Forecast')

    ax.set_title(f'Model Forecast vs. Actual Values for {currency} (Last {subset_size} Points)', fontsize=16)
    ax.set_xlabel('Time Step Index', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_test, predictions, currency):
    """
    Plots the distribution of the prediction errors. A good model should have
    errors centered around zero.
    """
    errors_median = y_test - predictions['median']
    errors_mean = y_test - predictions['mean']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.histplot(errors_median, ax=ax, color='darkorange', kde=True, label='Median Forecast Errors', bins=50)
    sns.histplot(errors_mean, ax=ax, color='green', kde=True, label='Mean Forecast Errors', bins=50, alpha=0.7)

    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    ax.set_title(f'Distribution of Prediction Errors for {currency}', fontsize=16)
    ax.set_xlabel('Error (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_prediction_vs_actual_scatter(y_test, predictions, currency):
    """
    Creates a scatter plot of predicted values vs. actual values.
    A perfect model would have all points on the 45-degree line.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(y_test, predictions['median'], alpha=0.5, label='Median Forecast', color='darkorange')
    ax.scatter(y_test, predictions['mean'], alpha=0.5, label='Mean Forecast', color='green')
    
    # Plot the "perfect prediction" line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Forecast Line')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title(f'Predicted vs. Actual Values for {currency}', fontsize=16)
    ax.set_xlabel('Actual Price', fontsize=12)
    ax.set_ylabel('Predicted Price', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# ==============================================================================
#  Main Execution Block
# ==============================================================================

def main():
    """Loads models and test data, then generates evaluation plots for each currency."""
    for currency in CURRENCIES_TO_PLOT:
        print(f"\n{'='*50}\nEvaluating and plotting for: {currency}\n{'='*50}")

        # --- 1. Load Test Data ---
        data_path = f'{PROCESSED_DATA_PATH}{currency}_master_dataset.parquet'
        print(f"Loading data from '{data_path}'...")
        try:
            master_df = pd.read_parquet(data_path)
        except FileNotFoundError:
            print(f"🔴 ERROR: Data file not found for {currency}. Skipping.")
            continue

        split_ratio = 0.8
        max_time = master_df['time'].max()
        split_point = int(max_time * split_ratio)
        test_df = master_df[master_df['time'] >= split_point].copy()

        # --- 2. Load Features and Models ---
        print("Loading saved models and feature list...")
        try:
            features = joblib.load(f'{MODELS_OUTPUT_PATH}feature_list.joblib')
            models = {
                'lower': lgb.Booster(model_file=f'{MODELS_OUTPUT_PATH}{currency}_lower_model.txt'),
                'median': lgb.Booster(model_file=f'{MODELS_OUTPUT_PATH}{currency}_median_model.txt'),
                'upper': lgb.Booster(model_file=f'{MODELS_OUTPUT_PATH}{currency}_upper_model.txt'),
                'mean': lgb.Booster(model_file=f'{MODELS_OUTPUT_PATH}{currency}_mean_model.txt'),
            }
        except Exception as e:
            print(f"🔴 ERROR: Could not load models for {currency}. Skipping. ({e})")
            continue

        # --- 3. Generate Predictions ---
        print("Generating predictions on the test set...")
        X_test = test_df[features]
        y_test = test_df['target']
        predictions = {n: m.predict(X_test) for n, m in models.items()}
        preds_df = pd.DataFrame(predictions, index=X_test.index)

        # --- 4. Create and Display Plots ---
        print("--- Generating Evaluation Plots ---")
        # plot_forecast_vs_actual(y_test, preds_df, currency)
        # plot_error_distribution(y_test, preds_df, currency)
        plot_prediction_vs_actual_scatter(y_test, preds_df, currency)
 
if __name__ == "__main__":
    main()