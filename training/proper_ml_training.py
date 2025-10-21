#!/usr/bin/env python3
"""
Complete ML Model Training Script
Usage: python proper_ml_training.py
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def load_and_prepare_data():
    """Load and prepare training data"""
    # Load file audit data
    file_data = []
    file_path = '../file_audit.json'
    if not os.path.exists(file_path):
        file_path = '../../file_audit.json'
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                file_data.append(record)
            except:
                continue
    
    # Load runtime data
    runtime_path = '../PDE-Runtimes-2025.csv'
    if not os.path.exists(runtime_path):
        runtime_path = '../../PDE-Runtimes-2025.csv'
    
    runtime_df = pd.read_csv(runtime_path)
    
    return file_data, runtime_df

def create_features(file_data, runtime_df):
    """Create feature matrix and targets"""
    # Aggregate file data by date/cycle
    daily_stats = {}
    
    for record in file_data:
        date_str = record.get('fil_creatn_dt', '')[:10]
        if not date_str:
            continue
            
        # Determine cycle from processing time
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
        
        # Temporal features
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
        
        # Add file count category
        fc = stats['file_count']
        feature_row['file_count_category'] = (
            0 if fc <= 2 else 1 if fc <= 5 else 2 if fc <= 10 else 3
        )
        
        # Add volume features
        feature_row['volume_per_file'] = stats['volume'] / fc if fc > 0 else 0
        feature_row['is_high_volume'] = 1 if stats['volume'] > 5000000 else 0
        
        features.append(feature_row)
        targets['file_count'].append(stats['file_count'])
        targets['volume'].append(stats['volume'])
        
        # Find matching runtime
        runtime_match = runtime_df[
            (pd.to_datetime(runtime_df['PDE date']).dt.strftime('%Y-%m-%d') == date_str) &
            (runtime_df['Cycle Type'] == cycle)
        ]
        
        if not runtime_match.empty:
            targets['runtime'].append(float(runtime_match.iloc[0]['Total Runtime']))
        else:
            targets['runtime'].append(None)
    
    return pd.DataFrame(features), targets

def train_file_count_model(X, y):
    """Train file count prediction model"""
    file_features = ['day_of_week', 'month', 'is_cycle1', 'is_weekend', 'is_monday', 'is_tuesday']
    X_file = X[file_features]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_file, y)
    
    return model, file_features

def train_volume_model(X, y):
    """Train volume prediction model"""
    volume_features = ['day_of_week', 'month', 'is_cycle1', 'is_weekend', 
                      'is_monday', 'is_tuesday', 'file_count', 'file_count_category']
    X_volume = X[volume_features]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_volume, y)
    
    return model, volume_features

def train_runtime_model(X, y):
    """Train runtime prediction model"""
    # Filter out None values
    valid_idx = [i for i, val in enumerate(y) if val is not None]
    X_runtime = X.iloc[valid_idx]
    y_runtime = [y[i] for i in valid_idx]
    
    runtime_features = ['day_of_week', 'month', 'is_cycle1', 'is_weekend', 
                       'is_monday', 'is_tuesday', 'file_count', 'file_count_category',
                       'volume', 'volume_per_file', 'is_high_volume']
    X_runtime = X_runtime[runtime_features]
    
    model = LassoCV(cv=5, random_state=42)
    model.fit(X_runtime, y_runtime)
    
    return model, runtime_features

def evaluate_model(model, X, y, model_name):
    """Evaluate model performance"""
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"{model_name} Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.3f}")
    print(f"  Sample predictions: {predictions[:5]}")
    print(f"  Sample actuals: {y[:5]}")
    print()

def main():
    """Main training pipeline"""
    print("Starting model retraining...")
    
    # Load data
    print("Loading data...")
    file_data, runtime_df = load_and_prepare_data()
    print(f"Loaded {len(file_data)} file records, {len(runtime_df)} runtime records")
    
    # Create features
    print("Creating features...")
    X, targets = create_features(file_data, runtime_df)
    print(f"Created {len(X)} training samples")
    
    # Train models
    print("Training file count model...")
    file_model, file_features = train_file_count_model(X, targets['file_count'])
    evaluate_model(file_model, X[file_features], targets['file_count'], "File Count")
    
    print("Training volume model...")
    volume_model, volume_features = train_volume_model(X, targets['volume'])
    evaluate_model(volume_model, X[volume_features], targets['volume'], "Volume")
    
    print("Training runtime model...")
    runtime_model, runtime_features = train_runtime_model(X, targets['runtime'])
    valid_runtime = [y for y in targets['runtime'] if y is not None]
    valid_X = X[[y is not None for y in targets['runtime']]][runtime_features]
    evaluate_model(runtime_model, valid_X, valid_runtime, "Runtime")
    
    # Save models
    print("Saving models...")
    
    # Save unified model
    unified_model = {
        'file_model': file_model,
        'volume_model': volume_model,
        'file_features': file_features,
        'volume_features': volume_features
    }
    joblib.dump(unified_model, '../lambda/enhanced_unified_model.pkl')
    
    # Save runtime model
    runtime_model_dict = {
        'model': runtime_model,
        'features': runtime_features
    }
    joblib.dump(runtime_model_dict, '../lambda/enhanced_runtime_model.pkl')
    
    print("Model retraining completed!")
    print("Files saved:")
    print("  - ../lambda/enhanced_unified_model.pkl")
    print("  - ../lambda/enhanced_runtime_model.pkl")

if __name__ == "__main__":
    main()