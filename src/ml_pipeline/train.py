import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================================================================
#  Configuration
# ==============================================================================
CURRENCIES_TO_TRAIN = ["CARD", "GARR", "HEST", "JUMP", "LOGN", "SIMP"]
PROCESSED_DATA_PATH = '../../data/processed/'
MODELS_OUTPUT_PATH = '../../models/'

# --- UPDATED: Define all model objectives ---
# We now have 3 quantile models and 1 mean model.
MODEL_CONFIGS = {
    'lower':  {'objective': 'quantile', 'alpha': 0.05, 'metric': 'quantile'},
    'median': {'objective': 'quantile', 'alpha': 0.5,  'metric': 'quantile'},
    'upper':  {'objective': 'quantile', 'alpha': 0.95, 'metric': 'quantile'},
    'mean':   {'objective': 'regression_l2', 'metric': 'l2'} # L2 loss is for the mean
}

# ==============================================================================
#  Main Training and Evaluation Function
# ==============================================================================

def train_and_evaluate_currency(currency: str):
    """
    Loads data, trains four models (lower, median, upper, mean),
    evaluates them, and saves the final artifacts.
    """
    print(f"\n{'='*50}\nStarting Training & Evaluation for: {currency}\n{'='*50}")

    # --- 1. Load Data ---
    data_path = f'{PROCESSED_DATA_PATH}{currency}_master_dataset.parquet'
    print(f"Step 1: Loading processed data from '{data_path}'...")
    try:
        master_df = pd.read_parquet(data_path)

    except FileNotFoundError:
        print(f"🔴 ERROR: Data file not found at '{data_path}'. Halting.")
        return

    # --- 2. Chronological Train-Test Split ---
    print("\nStep 2: Performing time-based train-test split...")
    last_idx = master_df[master_df['index'] == 399].last_valid_index() - 3

    train_df = master_df.iloc[:last_idx] 
    test_df = master_df.iloc[last_idx:]

    if train_df.empty or test_df.empty:
        print("🔴 ERROR: Split resulted in an empty training or testing set. Halting.")
        return

    # --- 3. Prepare Data for LightGBM (X, y) ---
    print("\nStep 3: Preparing feature (X) and target (y) sets...")
    TARGET_COLUMN = 'target'
    features = master_df.drop(columns='target').columns.tolist()
    print(len(features))
    # Display each column name alongside its dtype
    
    X_train, y_train = train_df[features], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[features], test_df[TARGET_COLUMN]
    # print(len(F'{X_train.columns = }'))
    from sklearn.model_selection import train_test_split
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False # Chronological split!
    )

    os.makedirs(MODELS_OUTPUT_PATH, exist_ok=True)
    feature_list_path = f'{MODELS_OUTPUT_PATH}feature_list.joblib'
    joblib.dump(features, feature_list_path)
    print(f"  - Saved {len(features)} feature names to '{feature_list_path}'")

    # --- 4. Train ALL Models ---
    print("\nStep 4: Training all four models...")
    trained_models = {}

    for name, config in MODEL_CONFIGS.items():
        print(f"  - Training '{name}' model (objective: {config['objective']})...")
        
        lgbm_params = {
            'objective': config['objective'],
            'metric': config['metric'],
            'n_estimators': 8000,           # more trees
            'learning_rate': 0.05,          # faster learning
            'num_leaves': 128,              # more complex leaves
            'max_depth': -1,                # no depth limit
            'feature_fraction': 1.0,        # use all features
            'bagging_fraction': 1.0,        # disable bagging
            'lambda_l1': 0.0,               # remove L1 regularization
            'lambda_l2': 0.0,               # remove L2 regularization
            'min_data_in_leaf': 20,         # smaller leaves allowed
            'n_jobs': -1,
            'seed': 42,
            'verbose': -1,
        }
        # Add alpha parameter only for quantile models
        if config['objective'] == 'quantile':
            lgbm_params['alpha'] = config['alpha']

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(
            X_train_sub, 
            y_train_sub,
            eval_set=[(X_val, y_val)],
            eval_metric=config['metric'],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)] # Stops after 50 rounds with no improvement
        )
        trained_models[name] = model
    
    print("  - Model training complete.")

    # --- 5. Evaluate on Test Set ---
    print("\nStep 5: Evaluating models on the unseen test set...")
    
    # Generate all predictions
    predictions = {name: model.predict(X_test) for name, model in trained_models.items()}

    # --- Evaluation for the Quantile Interval ---
    if name != 'mean':
        print("\n--- Probabilistic Forecast Evaluation ---")
        print("##############################################3")
        in_bounds = (y_test >= predictions['lower']) & (y_test <= predictions['upper'])
        coverage = np.mean(in_bounds)
        print(f"  - 📈 Prediction Interval Coverage: {coverage:.2%} (Target: ~90%)")
        
        avg_width = np.mean(predictions['upper'] - predictions['lower'])
        print(f"  - 📏 Average Interval Width: {avg_width:.2f}")

        # --- Evaluation for Point Forecasts (Mean vs. Median) ---
        print("\n--- Point Forecast Evaluation ---")
        mae_median = mean_absolute_error(y_test, predictions['median'])
        mae_mean = mean_absolute_error(y_test, predictions['mean'])
        print(f"  - 🎯 Median Forecast MAE: {mae_median:.2f}")
        print(f"  - 🎯 Mean Forecast MAE:   {mae_mean:.2f}")
        
        rmse_median = np.sqrt(mean_squared_error(y_test, predictions['median']))
        rmse_mean = np.sqrt(mean_squared_error(y_test, predictions['mean']))
        print(f"  - 🎯 Median Forecast RMSE: {rmse_median:.2f}")
        print(f"  - 🎯 Mean Forecast RMSE:   {rmse_mean:.2f}")

    import matplotlib.pyplot as plt

    # Step 5.1: Plot feature importances for each model
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, trained_models.items()):
        importances = model.feature_importances_
        # sort features by importance
        idx = np.arange(len(importances))
        print(importances)
        ax.barh([features[i] for i in idx], importances[idx], color='skyblue')
        ax.set_title(f"{name.capitalize()} Mode`l Feature Importance")
        ax.set_xlabel("Importance")
        ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plot_path = os.path.join(MODELS_OUTPUT_PATH, f"{currency}_feature_importances.png")
    plt.savefig(plot_path, dpi=300)
    print(f"  - Saved feature importance plot to '{plot_path}'")
    plt.close(fig)
    # --- 6. Save Final Models ---
    print("\nStep 6: Saving all final trained models...")
    for name, model in trained_models.items():
        model_path = f'{MODELS_OUTPUT_PATH}{currency}_{name}_model.txt'
        model.booster_.save_model(model_path)
        print(f"  - Saved model to '{model_path}'")

    print(f"\n✅ Workflow complete for {currency}.")

# ==============================================================================
#  Execution Block
# ==============================================================================

if __name__ == "__main__":
    for currency in CURRENCIES_TO_TRAIN:
        train_and_evaluate_currency(currency)