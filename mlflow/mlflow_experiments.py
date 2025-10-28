#!/usr/bin/env python3
"""
MLflow Model Experiments for Prediction Accuracy
Usage: python mlflow_experiments.py
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import warnings

# Suppress MLflow warnings
warnings.filterwarnings('ignore')
os.environ['MLFLOW_DISABLE_ENV_CREATION'] = 'true'

# Set MLflow tracking URI (local)
mlflow.set_tracking_uri("file:./mlruns")

def load_training_data():
    """Load and prepare training data"""
    # Load file audit data
    file_data = []
    file_path = '../lambda/file_audit.json'
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                file_data.append(record)
            except:
                continue
    
    # Load runtime data
    runtime_df = pd.read_csv('../lambda/PDE-Runtimes-2025.csv')
    
    return file_data, runtime_df

def create_features(file_data, runtime_df):
    """Create feature matrix and targets"""
    daily_stats = {}
    
    for record in file_data:
        date_str = record.get('fil_creatn_dt', '')[:10]
        if not date_str:
            continue
            
        time_str = str(record.get('proc_strt_time', '0'))
        hour = int(time_str[:2]) if len(time_str) >= 2 else 12
        cycle = 'Cycle1' if hour < 12 else 'Cycle3'
        
        key = (date_str, cycle)
        if key not in daily_stats:
            daily_stats[key] = {'file_count': 0, 'volume': 0}
        
        daily_stats[key]['file_count'] += 1
        daily_stats[key]['volume'] += int(record.get('in_tot_rec_cnt', 0))
    
    # Create feature matrix
    features = []
    targets = {'file_count': [], 'volume': [], 'runtime': []}
    
    for (date_str, cycle), stats in daily_stats.items():
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        
        feature_row = {
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'is_cycle1': 1 if cycle == 'Cycle1' else 0,
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'is_monday': 1 if dt.weekday() == 0 else 0,
            'is_tuesday': 1 if dt.weekday() == 1 else 0,
            'file_count': stats['file_count'],
            'volume': stats['volume']
        }
        
        fc = stats['file_count']
        feature_row['file_count_category'] = (
            0 if fc <= 2 else 1 if fc <= 5 else 2 if fc <= 10 else 3
        )
        feature_row['volume_per_file'] = stats['volume'] / fc if fc > 0 else 0
        feature_row['is_high_volume'] = 1 if stats['volume'] > 5000000 else 0
        
        features.append(feature_row)
        targets['file_count'].append(stats['file_count'])
        targets['volume'].append(stats['volume'])
        
        # Find matching runtime
        try:
            runtime_match = runtime_df[
                (pd.to_datetime(runtime_df['PDE date'], dayfirst=True).dt.strftime('%Y-%m-%d') == date_str) &
                (runtime_df['Cycle Type'] == cycle)
            ]
            
            if not runtime_match.empty:
                targets['runtime'].append(float(runtime_match.iloc[0]['Total Runtime']))
            else:
                targets['runtime'].append(None)
        except:
            targets['runtime'].append(None)
    
    return pd.DataFrame(features), targets

def experiment_file_count_models(X, y):
    """Experiment with different models for file count prediction"""
    file_features = ['day_of_week', 'month', 'is_cycle1', 'is_weekend', 'is_monday', 'is_tuesday']
    X_file = X[file_features]
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Lasso': LassoCV(cv=5, random_state=42),
        'Ridge': RidgeCV(cv=5),
        'ElasticNet': ElasticNetCV(cv=5, random_state=42)
    }
    
    results = {}
    
    with mlflow.start_run(run_name="file_count_experiments"):
        for name, model in models.items():
            with mlflow.start_run(run_name=f"file_count_{name}", nested=True):
                # Train model
                model.fit(X_file, y)
                
                # Make predictions
                y_pred = model.predict(X_file)
                
                # Calculate metrics
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Log parameters
                mlflow.log_param("model_type", name)
                mlflow.log_param("features", file_features)
                mlflow.log_param("n_samples", len(y))
                
                # Log metrics
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                
                # Create input example
                input_example = X_file.iloc[:1]
                
                # Log model with signature
                mlflow.sklearn.log_model(
                    model, 
                    name=f"file_count_{name}_model",
                    input_example=input_example
                )
                
                # Create and log prediction plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 6))
                plt.scatter(y, y_pred, alpha=0.6)
                plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
                plt.xlabel('Actual File Count')
                plt.ylabel('Predicted File Count')
                plt.title(f'File Count Predictions - {name}')
                plt.savefig(f'file_count_{name}_plot.png')
                mlflow.log_artifact(f'file_count_{name}_plot.png')
                plt.close()
                
                # Log feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': file_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    importance_df.to_csv(f'file_count_{name}_importance.csv', index=False)
                    mlflow.log_artifact(f'file_count_{name}_importance.csv')
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'r2': r2
                }
                
                print(f"File Count - {name}: MAE={mae:.2f}, R²={r2:.3f}")
    
    return results

def experiment_volume_models(X, y):
    """Experiment with different models for volume prediction"""
    volume_features = ['day_of_week', 'month', 'is_cycle1', 'is_weekend', 
                      'is_monday', 'is_tuesday', 'file_count', 'file_count_category']
    X_volume = X[volume_features]
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Lasso': LassoCV(cv=5, random_state=42),
        'Ridge': RidgeCV(cv=5),
        'ElasticNet': ElasticNetCV(cv=5, random_state=42)
    }
    
    results = {}
    
    with mlflow.start_run(run_name="volume_experiments"):
        for name, model in models.items():
            with mlflow.start_run(run_name=f"volume_{name}", nested=True):
                # Train model
                model.fit(X_volume, y)
                
                # Make predictions
                y_pred = model.predict(X_volume)
                
                # Calculate metrics
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Log parameters
                mlflow.log_param("model_type", name)
                mlflow.log_param("features", volume_features)
                mlflow.log_param("n_samples", len(y))
                
                # Log metrics
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                
                # Create input example
                input_example = X_volume.iloc[:1]
                
                # Log model with signature
                mlflow.sklearn.log_model(
                    model, 
                    name=f"volume_{name}_model",
                    input_example=input_example
                )
                
                # Create and log prediction plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 6))
                plt.scatter(y, y_pred, alpha=0.6)
                plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
                plt.xlabel('Actual Volume')
                plt.ylabel('Predicted Volume')
                plt.title(f'Volume Predictions - {name}')
                plt.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
                plt.savefig(f'volume_{name}_plot.png')
                mlflow.log_artifact(f'volume_{name}_plot.png')
                plt.close()
                
                # Log feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': volume_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    importance_df.to_csv(f'volume_{name}_importance.csv', index=False)
                    mlflow.log_artifact(f'volume_{name}_importance.csv')
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'r2': r2
                }
                
                print(f"Volume - {name}: MAE={mae:,.0f}, R²={r2:.3f}")
    
    return results

def experiment_runtime_models(X, y):
    """Experiment with different models for runtime prediction"""
    # Filter out None values
    valid_idx = [i for i, val in enumerate(y) if val is not None]
    X_runtime = X.iloc[valid_idx]
    y_runtime = [y[i] for i in valid_idx]
    
    if len(y_runtime) == 0:
        print("No runtime data available for experiments")
        return {}
    
    runtime_features = ['day_of_week', 'month', 'is_cycle1', 'is_weekend', 
                       'is_monday', 'is_tuesday', 'file_count', 'file_count_category',
                       'volume', 'volume_per_file', 'is_high_volume']
    X_runtime = X_runtime[runtime_features]
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Lasso': LassoCV(cv=5, random_state=42),
        'Ridge': RidgeCV(cv=5),
        'ElasticNet': ElasticNetCV(cv=5, random_state=42)
    }
    
    results = {}
    
    with mlflow.start_run(run_name="runtime_experiments"):
        for name, model in models.items():
            with mlflow.start_run(run_name=f"runtime_{name}", nested=True):
                # Train model
                model.fit(X_runtime, y_runtime)
                
                # Make predictions
                y_pred = model.predict(X_runtime)
                
                # Calculate metrics
                mae = mean_absolute_error(y_runtime, y_pred)
                mse = mean_squared_error(y_runtime, y_pred)
                r2 = r2_score(y_runtime, y_pred)
                
                # Log parameters
                mlflow.log_param("model_type", name)
                mlflow.log_param("features", runtime_features)
                mlflow.log_param("n_samples", len(y_runtime))
                
                # Log metrics
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                
                # Create input example
                input_example = X_runtime.iloc[:1]
                
                # Log model with signature
                mlflow.sklearn.log_model(
                    model, 
                    name=f"runtime_{name}_model",
                    input_example=input_example
                )
                
                # Create and log prediction plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 6))
                plt.scatter(y_runtime, y_pred, alpha=0.6)
                plt.plot([min(y_runtime), max(y_runtime)], [min(y_runtime), max(y_runtime)], 'r--', lw=2)
                plt.xlabel('Actual Runtime (minutes)')
                plt.ylabel('Predicted Runtime (minutes)')
                plt.title(f'Runtime Predictions - {name}')
                plt.savefig(f'runtime_{name}_plot.png')
                mlflow.log_artifact(f'runtime_{name}_plot.png')
                plt.close()
                
                # Log feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': runtime_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    importance_df.to_csv(f'runtime_{name}_importance.csv', index=False)
                    mlflow.log_artifact(f'runtime_{name}_importance.csv')
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'r2': r2
                }
                
                print(f"Runtime - {name}: MAE={mae:.1f}min, R²={r2:.3f}")
    
    return results

def compare_models(file_results, volume_results, runtime_results):
    """Compare model performance and select best ones"""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Best file count model
    best_file = min(file_results.items(), key=lambda x: x[1]['mae'])
    print(f"Best File Count Model: {best_file[0]} (MAE: {best_file[1]['mae']:.2f})")
    
    # Best volume model
    best_volume = min(volume_results.items(), key=lambda x: x[1]['mae'])
    print(f"Best Volume Model: {best_volume[0]} (MAE: {best_volume[1]['mae']:,.0f})")
    
    # Best runtime model
    if runtime_results:
        best_runtime = min(runtime_results.items(), key=lambda x: x[1]['mae'])
        print(f"Best Runtime Model: {best_runtime[0]} (MAE: {best_runtime[1]['mae']:.1f}min)")
    
    # Log comparison experiment
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_param("best_file_model", best_file[0])
        mlflow.log_param("best_volume_model", best_volume[0])
        if runtime_results:
            mlflow.log_param("best_runtime_model", best_runtime[0])
        
        mlflow.log_metric("best_file_mae", best_file[1]['mae'])
        mlflow.log_metric("best_volume_mae", best_volume[1]['mae'])
        if runtime_results:
            mlflow.log_metric("best_runtime_mae", best_runtime[1]['mae'])

def main():
    """Main experiment pipeline"""
    print("Starting MLflow Model Experiments...")
    print("="*50)
    
    # Set experiment
    mlflow.set_experiment("ML_Prediction_Models")
    
    # Load data
    print("Loading training data...")
    file_data, runtime_df = load_training_data()
    print(f"Loaded {len(file_data)} file records")
    
    # Create features
    print("Creating features...")
    X, targets = create_features(file_data, runtime_df)
    print(f"Created {len(X)} training samples")
    
    # Run experiments
    print("\nRunning File Count Experiments...")
    file_results = experiment_file_count_models(X, targets['file_count'])
    
    print("\nRunning Volume Experiments...")
    volume_results = experiment_volume_models(X, targets['volume'])
    
    print("\nRunning Runtime Experiments...")
    runtime_results = experiment_runtime_models(X, targets['runtime'])
    
    # Compare results
    compare_models(file_results, volume_results, runtime_results)
    
    print(f"\n✅ Experiments completed!")
    print(f"View results: mlflow ui --backend-store-uri file:./mlruns")

if __name__ == "__main__":
    main()