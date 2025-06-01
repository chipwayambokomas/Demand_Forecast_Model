import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit # MVP_CHANGE: Added for HPO
import os
import joblib
from typing import Optional, List, Dict, Any

# -----------------------------
# Configuration
# -----------------------------
SALES_INVENTORY_CSV = "data/sales_inventory_data.csv"
PO_CSV = "data/incoming_orders_data.csv"
OUTPUT_FORECAST_CSV = "forecasts/fnsku_recommended_orders.csv"
MODEL_FILENAME = "saved_model/demand_forecast_model_hpo.joblib"

# --- User Choices ---
FORCE_RETRAIN_MODEL: bool = True 
ENABLE_HPO: bool = True 
DEFAULT_FORECAST_HORIZON_WEEKS: int = 16
MODEL_PREDICTION_CHUNK_WEEKS: int = 4
# --- End User Choices ---

# --- HPO Configuration (if ENABLE_HPO is True) ---
# Number of splits for TimeSeriesSplit during HPO.
HPO_CV_SPLITS: int = 3 # Keep low for faster MVP runs
# Parameter grid for LightGBM. 
LGBM_PARAM_GRID: Dict[str, List[Any]] = {
    'n_estimators': [50, 100, 150],       # Number of trees
    'learning_rate': [0.05, 0.1],         # Step size shrinkage
    'max_depth': [4, 6, 8],               # Max depth of individual trees
    'num_leaves': [20, 31, 40],           # Max number of leaves in one tree (<= 2^max_depth)
}
# Scoring metric for HPO. 
HPO_SCORING_METRIC: str = 'neg_mean_absolute_error'
# --- End HPO Configuration ---


TARGET_VARIABLE = 'target_sales_4w'
MODEL_FEATURES = [
    'inventory_level', 'stockout_flag', 'days_of_supply', 'incoming_stock',
    'rolling_sales_4w', 'rolling_sales_8w', 'rolling_sales_12w', 'rolling_sales_26w',
    'lag_sales_1w', 'lag_sales_4w', 'lag_sales_12w',
    'trend_4w_vs_8w', 'trend_8w_vs_26w',
    'week_of_year', 'month', 'quarter'
]

# -----------------------------
# 1. Load and prepare the data
# -----------------------------
def load_and_prepare_data(sales_inventory_csv: str, po_csv: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(sales_inventory_csv):
        print(f"Warning: Sales inventory file not found at '{sales_inventory_csv}'")
        return None
    if po_csv and not os.path.exists(po_csv):
        print(f"Creating a dummy {po_csv} for demonstration...")
        return None

    df = pd.read_csv(sales_inventory_csv)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.sort_values(['FNSKU', 'Date'])
    
    #create year-week and week_start columns
    df['year_week'] = df['Date'].dt.strftime('%Y-W%U')
    df['week_start'] = df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='d')
    
    #group by FNSKU, year_week, and week_start and add up units_sold and average inventory_level for each week and turn this into a dataframe of its own
    df_weekly = df.groupby(['FNSKU', 'year_week', 'week_start']).agg({
        'units_sold': 'sum',
        'inventory_level': 'mean'
    }).reset_index()
    
    df_weekly = df_weekly.sort_values(['FNSKU', 'week_start'])
    
    if df_weekly.empty:
        print("Warning: Weekly dataframe is empty after aggregation.")
        return pd.DataFrame()

    print(f"Weekly data range: {df_weekly['week_start'].min().date()} to {df_weekly['week_start'].max().date()}")
    print(f"Number of unique FNSKUs: {df_weekly['FNSKU'].nunique()}")

    # Add new columns to df_weekly DataFrame based on the 'week_start' datetime column

    # Extract the ISO week number from 'week_start' and convert it to integer
    df_weekly['week_of_year'] = df_weekly['week_start'].dt.isocalendar().week.astype(int)

    # Extract the month number from 'week_start'
    df_weekly['month'] = df_weekly['week_start'].dt.month

    # Extract the quarter (1-4) from 'week_start'
    df_weekly['quarter'] = df_weekly['week_start'].dt.quarter

    # Extract the year from 'week_start'
    df_weekly['year'] = df_weekly['week_start'].dt.year

    # Calculate rolling mean of past sales for each FNSKU (excluding current week)
    for N in [4, 8, 12, 26]:
        df_weekly[f'rolling_sales_{N}w'] = df_weekly.groupby(['FNSKU'])['units_sold'].transform(
            lambda x: x.shift(1).rolling(N, min_periods=max(1, N//2)).mean()
        )
    # Create lag features for sales (previous 1, 4, and 12 weeks)
    for N in [1, 4, 12]:
        df_weekly[f'lag_sales_{N}w'] = df_weekly.groupby(['FNSKU'])['units_sold'].shift(N)

    # Year-over-year sales: previous year's sales for the same week number
    #df_weekly['sales_same_week_last_year'] = df_weekly.groupby(['FNSKU', 'week_of_year'])['units_sold'].shift(1)
    # Year-over-year growth: percent change vs. last year's same week
    #df_weekly['yoy_growth'] = (df_weekly['units_sold'] / (df_weekly['sales_same_week_last_year'] + 1e-5)) - 1
    # Short-term trend: 4-week vs. 8-week rolling sales
    df_weekly['trend_4w_vs_8w'] = (df_weekly['rolling_sales_4w'] / (df_weekly['rolling_sales_8w'] + 1e-5)) - 1
    # Medium-term trend: 8-week vs. 26-week rolling sales
    df_weekly['trend_8w_vs_26w'] = (df_weekly['rolling_sales_8w'] / (df_weekly['rolling_sales_26w'] + 1e-5)) - 1
    # Stockout flag: 1 if inventory is zero or negative, else 0
    df_weekly['stockout_flag'] = (df_weekly['inventory_level'] <= 0).astype(int)
    # Average daily demand over last 4 weeks
    df_weekly['daily_demand_4w'] = df_weekly['rolling_sales_4w'] / 7
    # Days of supply: inventory divided by average daily demand
    df_weekly['days_of_supply'] = df_weekly['inventory_level'] / (df_weekly['daily_demand_4w'] + 1e-5)
    # Target variable: sales in the next forecast chunk (e.g., next 4 weeks)
    df_weekly[TARGET_VARIABLE] = df_weekly.groupby(['FNSKU'])['units_sold'].shift(-MODEL_PREDICTION_CHUNK_WEEKS)

    if po_csv and os.path.exists(po_csv):
        try:
            df_po = pd.read_csv(po_csv)
            if not df_po.empty and all(col in df_po.columns for col in ['FNSKU', 'Expected Arrival', 'Units']):
                df_po[['week_num_str', 'year_str']] = df_po['Expected Arrival'].astype(str).str.split('/', expand=True)
                df_po['week_num'] = pd.to_numeric(df_po['week_num_str'], errors='coerce')
                df_po['year'] = pd.to_numeric(df_po['year_str'], errors='coerce')
                df_po.dropna(subset=['week_num', 'year'], inplace=True)

                if not df_po.empty:
                    df_po['week_num'] = df_po['week_num'].astype(int)
                    df_po['year'] = df_po['year'].astype(int)
                    df_po['iso_week_format'] = df_po['year'].astype(str) + '-W' + df_po['week_num'].astype(str).str.zfill(2) + '-1'
                    df_po['expected_delivery_date'] = pd.to_datetime(df_po['iso_week_format'], format='%G-W%V-%u', errors='coerce')
                    df_po.dropna(subset=['expected_delivery_date'], inplace=True)

                    df_po_sum = df_po.groupby(['FNSKU', 'expected_delivery_date'])['Units'].sum().reset_index()
                    df_po_sum.rename(columns={'expected_delivery_date': 'week_start'}, inplace=True)
                    
                    df_weekly = df_weekly.merge(df_po_sum, how='left', on=['FNSKU', 'week_start'])
                    df_weekly['incoming_stock'] = df_weekly['Units'].fillna(0)
                    if 'Units' in df_weekly.columns: del df_weekly['Units']
                    print("Purchase order data integrated.")
                else:
                    print("PO data empty/unparseable. 'incoming_stock' = 0.")
                    df_weekly['incoming_stock'] = 0
            else:
                print(f"PO file '{po_csv}' empty/missing cols. 'incoming_stock' = 0.")
                df_weekly['incoming_stock'] = 0
        except Exception as e:
            print(f"Error processing PO data: {e}. 'incoming_stock' = 0.")
            df_weekly['incoming_stock'] = 0
    else:
        df_weekly['incoming_stock'] = 0
        if po_csv: print(f"PO file '{po_csv}' not found. 'incoming_stock' = 0.")
        else: print("No PO data. 'incoming_stock' = 0.")
    
    return df_weekly.copy()

# -----------------------------
# 2. Train or Load ML model
# -----------------------------
def train_or_load_model(
    df_train_data: pd.DataFrame, 
    model_features: List[str], 
    target_col: str, 
    model_path: str, 
    force_retrain: bool = False,
    enable_hpo: bool = False 
) -> Optional[LGBMRegressor]:
    if not force_retrain and os.path.exists(model_path):
        try:
            print(f"Loading pre-trained model from '{model_path}'...")
            model = joblib.load(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Retraining.")

    print("Processing data for model training...")
    df_model = df_train_data.dropna(subset=model_features + [target_col]).copy()
    
    if len(df_model) < 50: # Check after NaN drop
        print(f"Error: Insufficient data ({len(df_model)} samples) for training after NaN removal.")
        return None
    
    # Data must be sorted by time for TimeSeriesSplit and chronological evaluation
    df_model_sorted = df_model.sort_values('week_start').reset_index(drop=True)
    X = df_model_sorted[model_features]
    y = df_model_sorted[target_col]
    
    print(f"Training on {len(X)} samples. Original weekly records: {len(df_train_data)}.")
    
    # Chronological split for final evaluation (if not doing HPO or after HPO)
    # If HPO is enabled, GridSearchCV handles its own internal CV splits.
    # We still define a hold-out test set here for a final check of the *best* HPO model.
    if len(X) < 10: # Too small for a meaningful split
        X_train, y_train = X, y
        X_test, y_test = pd.DataFrame(columns=model_features), pd.Series(dtype=float)
        print("Warning: Dataset too small for train/test split. Using all data for training.")
    else:
        test_size_ratio = 0.2 # e.g., last 20% of the (time-sorted) data for testing
        split_index = int(len(X) * (1 - test_size_ratio))
        X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
        X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]
        print(f"Train size: {len(X_train)}, Test size (final hold-out): {len(X_test)}")

    if X_train.empty:
        print("Error: Training set empty after split. Cannot train.")
        return None

    best_model: Optional[LGBMRegressor] = None

    if enable_hpo:
        print("\n--- Hyperparameter Optimization (HPO) Enabled ---")
        if len(X_train) < HPO_CV_SPLITS + 1 : # TimesSeriesSplit needs n_samples > n_splits
             print(f"Warning: Training data size ({len(X_train)}) is too small for HPO with {HPO_CV_SPLITS} CV splits. Skipping HPO.")
             enable_hpo = False # Fallback to default training

    if enable_hpo:
        print(f"Starting GridSearchCV with {HPO_CV_SPLITS} time-series splits...")
        print(f"Parameter grid: {LGBM_PARAM_GRID}")
        
        # TimeSeriesSplit for HPO cross-validation
        # max_train_size can be set to limit the size of the training fold in each split,
        # useful for very long time series to speed up HPO.
        tscv = TimeSeriesSplit(n_splits=HPO_CV_SPLITS) #, max_train_size=None
        
        # Base estimator
        base_lgbm = LGBMRegressor(random_state=42, verbosity=-1)
        
        grid_search = GridSearchCV(
            estimator=base_lgbm,
            param_grid=LGBM_PARAM_GRID,
            scoring=HPO_SCORING_METRIC,
            cv=tscv,
            verbose=1, # Set to 2 or 3 for more detailed output during HPO
            n_jobs=-1  # Use all available CPU cores
        )
        
        try:
            grid_search.fit(X_train, y_train) # HPO is done on the training part of the chronological split
            print("\nHPO Results:")
            print(f"  Best Parameters: {grid_search.best_params_}")
            print(f"  Best CV Score ({HPO_SCORING_METRIC}): {grid_search.best_score_:.4f}")
            best_model = grid_search.best_estimator_
        except Exception as e:
            print(f"Error during GridSearchCV: {e}. Training with default parameters instead.")
            # Fallback to default model if HPO fails
            best_model = LGBMRegressor(random_state=42, verbosity=-1, n_estimators=100, learning_rate=0.1, max_depth=6)
            best_model.fit(X_train, y_train)
            
    else: # HPO is not enabled or skipped
        print("\nTraining model with default parameters (HPO disabled or skipped)...")
        best_model = LGBMRegressor(random_state=42, verbosity=-1, n_estimators=100, learning_rate=0.1, max_depth=6)
        best_model.fit(X_train, y_train)

    # Evaluate the chosen model (either from HPO or default) on the hold-out test set
    if best_model and not X_test.empty and not y_test.empty:
        print("\nEvaluating final model on the hold-out test set...")
        y_pred_test = best_model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        y_test_safe = y_test.copy()
        y_test_safe[y_test_safe == 0] = 1e-5 # Avoid division by zero for MAPE
        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test_safe)) * 100
        
        print(f"  Test Set MAE: {mae_test:.2f}")
        print(f"  Test Set MAPE: {mape_test:.1f}%")
    elif not best_model:
        print("No model was trained successfully.")
        return None
    else:
        print("Test set was empty, skipping final evaluation on hold-out data.")

    if best_model:
        try:
            # Display feature importance from the final best model
            if hasattr(best_model, 'feature_importances_'):
                feature_importance_df = pd.DataFrame({
                    'feature': X_train.columns, # Use columns from X_train
                    'importance': best_model.feature_importances_
                })
                print("\nTop 10 Features (from final model):\n", feature_importance_df.sort_values('importance', ascending=False).head(10))

            joblib.dump(best_model, model_path)
            print(f"Final model saved to '{model_path}'")
        except Exception as e:
            print(f"Error saving final model: {e}")
            
    return best_model

# -----------------------------
# 3. Forecast & order planning
# -----------------------------
def forecast_and_plan( 
    df_weekly_all_history: pd.DataFrame, 
    model: LGBMRegressor, 
    features_for_prediction: List[str], # This is MODEL_FEATURES
    forecast_horizon_weeks: int, # Total weeks to plan for
    model_predicts_n_weeks: int # How many weeks the model directly predicts (MODEL_PREDICTION_CHUNK_WEEKS)
) -> pd.DataFrame:
    """
    Generate sales forecasts by predicting a chunk and scaling, then recommend order quantities.
    Order recommendation is: Forecasted Demand - Current Inventory.
    """
    if df_weekly_all_history.empty:
        print("Cannot generate forecasts: input historical data is empty.")
        return pd.DataFrame()
        
    df_weekly_all_history['week_start'] = pd.to_datetime(df_weekly_all_history['week_start'])
    df_latest = df_weekly_all_history.loc[df_weekly_all_history.groupby('FNSKU')['week_start'].idxmax()]

    if df_latest.empty:
        print("No latest data found for any FNSKU. Cannot generate forecasts.")
        return pd.DataFrame()

    forecast_results_list = []
    successful_forecasts = 0
    
    print(f"\nGenerating forecasts for {len(df_latest)} FNSKUs using simple scaling...")

    for _, row in df_latest.iterrows():
        fnsku = row['FNSKU']
        current_inventory = row['inventory_level'] 
        last_data_week = row['week_start'].date()

        feature_values = row[features_for_prediction]
        
        # MVP: Simple NaN check - skip if any feature needed for prediction is NaN
        if feature_values.isnull().any():
            nan_features = feature_values[feature_values.isnull()].index.tolist()
            print(f"  Skipping FNSKU {fnsku} (last data: {last_data_week}): missing required features: {', '.join(nan_features)}. "
                  f"Total NaNs: {feature_values.isnull().sum()}.")
            continue
        
        X_pred = feature_values.values.reshape(1, -1)
        
        try:
            # Model predicts sales for 'model_predicts_n_weeks' (e.g., 4 weeks)
            forecast_sales_model_chunk = model.predict(X_pred)[0]
            forecast_sales_model_chunk = max(0, forecast_sales_model_chunk) 
        except Exception as e:
            print(f"  Error predicting for FNSKU {fnsku} (last data: {last_data_week}): {e}. Skipping.")
            continue
            
        # Scale the model's prediction to the full forecast_horizon_weeks
        if model_predicts_n_weeks <= 0: # Avoid division by zero
            print(f"  Warning for FNSKU {fnsku}: model_predicts_n_weeks is {model_predicts_n_weeks}, cannot scale. Assuming 0 total forecast.")
            total_forecast_over_horizon = 0.0
        else:
            total_forecast_over_horizon = forecast_sales_model_chunk * (forecast_horizon_weeks / float(model_predicts_n_weeks))
            total_forecast_over_horizon = max(0, total_forecast_over_horizon)

        recommended_order_qty = total_forecast_over_horizon - current_inventory
        recommended_order_qty = max(0, recommended_order_qty)
        
        forecast_results_list.append({
            'FNSKU': fnsku,
            'last_data_week': last_data_week,
            'current_inventory': round(current_inventory, 2),
            'days_of_supply_current': round(row['days_of_supply'],1) if pd.notnull(row['days_of_supply']) else 'N/A',
            f'forecast_sales_{model_predicts_n_weeks}w_chunk': round(forecast_sales_model_chunk, 2), # Show what model predicted
            f'forecast_sales_total_{forecast_horizon_weeks}w': round(total_forecast_over_horizon, 2),
            'recommended_order_qty': round(recommended_order_qty, 0)
        })
        successful_forecasts += 1

    if not forecast_results_list:
        print("No forecasts were successfully generated.")
        return pd.DataFrame()

    df_forecast = pd.DataFrame(forecast_results_list)
    print(f"\nSuccessfully generated forecasts for {successful_forecasts} FNSKUs.")
    
    if not df_forecast.empty:
        print(f"\nForecast Summary (for {forecast_horizon_weeks} weeks horizon):")
        print(f"  Average {forecast_horizon_weeks}-week total forecast per FNSKU: {df_forecast[f'forecast_sales_total_{forecast_horizon_weeks}w'].mean():.1f} units")
        print(f"  Total recommended order units: {df_forecast['recommended_order_qty'].sum():.0f} units")
        dos_numeric = pd.to_numeric(df_forecast['days_of_supply_current'], errors='coerce')
        print(f"  FNSKUs with low current inventory (<7 days supply): {(dos_numeric < 7).sum()}")
    
    return df_forecast
# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    print("--- Demand Forecasting (MVP v4 - HPO, Recursive) ---")
    
    print("\nStep 1: Loading and preparing data...")
    df_weekly_prepared = load_and_prepare_data(SALES_INVENTORY_CSV, PO_CSV)

    if df_weekly_prepared.empty or len(df_weekly_prepared) < 20: # Need more data for HPO/splits
        print("\nNot enough historical data. Exiting.")
    else:
        print(f"\nStep 2: Training or loading model (Force Retrain: {FORCE_RETRAIN_MODEL}, HPO: {ENABLE_HPO})...")
        ml_model = train_or_load_model(
            df_weekly_prepared, 
            MODEL_FEATURES, 
            TARGET_VARIABLE, 
            MODEL_FILENAME, 
            force_retrain=FORCE_RETRAIN_MODEL,
            enable_hpo=ENABLE_HPO # Pass the HPO flag
        )
        
        if ml_model:
            print(f"\nStep 3: Generating forecasts for {DEFAULT_FORECAST_HORIZON_WEEKS} weeks...")
            forecast_results_df = forecast_and_plan(
                df_weekly_prepared, 
                ml_model, 
                MODEL_FEATURES, 
                forecast_horizon_weeks=DEFAULT_FORECAST_HORIZON_WEEKS,
                model_predicts_n_weeks = MODEL_PREDICTION_CHUNK_WEEKS
            )
            
            if not forecast_results_df.empty:
                try:
                    forecast_results_df.to_csv(OUTPUT_FORECAST_CSV, index=False)
                    print(f"\nResults saved to '{OUTPUT_FORECAST_CSV}'")
                    print(f"\nSample Forecasts (from '{OUTPUT_FORECAST_CSV}'):")
                    print(forecast_results_df.head())
                except Exception as e:
                    print(f"Error saving forecast results: {e}")
            else:
                print("\nNo forecast results generated.")
        else:
            print("\nModel not available. Skipping forecasting.")
            
    print("\n--- Script execution finished. ---")