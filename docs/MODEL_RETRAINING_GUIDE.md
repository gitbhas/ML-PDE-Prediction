# Model Retraining Guide

## Overview
This guide provides step-by-step instructions for retraining ML models with updated datasets.

## Retraining Process

### 1. Data Update
```bash
# Download latest data from S3 (if applicable)
aws s3 cp s3://ddps-prd-oes-staging/FILEAU.ML.EXT000.json file_audit.json

# Or update local files with new data
# Append new records to existing file_audit.json
# Update PDE-Runtimes-2025.csv with recent runtime measurements
```

### 2. Training Script Execution
```bash
# Run comprehensive model training
python proper_ml_training.py

# Or run specific training scripts
python train_all_models.py
python comprehensive_model_training.py
```

### 3. Training Script Structure (`proper_ml_training.py`)
```python
import pandas as pd
import numpy as np
import json
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
    with open('file_audit.json', 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                file_data.append(record)
            except:
                continue
    
    # Load runtime data
    runtime_df = pd.read_csv('PDE-Runtimes-2025.csv')
    
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
    joblib.dump(unified_model, 'enhanced_unified_model.pkl')
    
    # Save runtime model
    runtime_model_dict = {
        'model': runtime_model,
        'features': runtime_features
    }
    joblib.dump(runtime_model_dict, 'enhanced_runtime_model.pkl')
    
    print("Model retraining completed!")
    print("Files saved:")
    print("  - enhanced_unified_model.pkl")
    print("  - enhanced_runtime_model.pkl")

if __name__ == "__main__":
    main()
```

### 4. Model Validation Script
```python
# validate_models.py
import joblib
import pandas as pd
from datetime import datetime, timedelta

def validate_predictions():
    """Validate new models against recent data"""
    # Load new models
    unified_model = joblib.load('enhanced_unified_model.pkl')
    runtime_model = joblib.load('enhanced_runtime_model.pkl')
    
    # Test predictions for recent dates
    test_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                  for i in range(1, 8)]
    
    for date in test_dates:
        for cycle in ['Cycle1', 'Cycle3']:
            # Create test features
            dt = datetime.strptime(date, '%Y-%m-%d')
            if dt.weekday() == 6 or (dt.weekday() == 5 and cycle == 'Cycle3'):
                continue
                
            features = create_prediction_features(date, cycle)
            
            # Make predictions
            file_pred = unified_model['file_model'].predict([features['file']])[0]
            volume_pred = unified_model['volume_model'].predict([features['volume']])[0]
            runtime_pred = runtime_model['model'].predict([features['runtime']])[0]
            
            print(f"{date} {cycle}: Files={file_pred:.0f}, Volume={volume_pred:,.0f}, Runtime={runtime_pred:.1f}min")

if __name__ == "__main__":
    validate_predictions()
```

### 5. Deployment Update
```bash
# Copy new models to Lambda directory
cp enhanced_unified_model.pkl cdkapp/lambda/
cp enhanced_runtime_model.pkl cdkapp/lambda/

# Redeploy Lambda function
cd cdkapp
npx cdk deploy PredictionLambdaStack --app "python prediction_app.py" --require-approval never
```

### 6. Automated Retraining Pipeline
```bash
#!/bin/bash
# retrain_pipeline.sh

echo "Starting automated retraining pipeline..."

# Step 1: Backup current models
cp enhanced_unified_model.pkl enhanced_unified_model_backup.pkl
cp enhanced_runtime_model.pkl enhanced_runtime_model_backup.pkl

# Step 2: Download latest data (if from S3)
# aws s3 cp s3://your-bucket/file_audit.json .

# Step 3: Run training
python proper_ml_training.py

# Step 4: Validate models
python validate_models.py

# Step 5: Deploy if validation passes
if [ $? -eq 0 ]; then
    echo "Validation passed, deploying new models..."
    cp enhanced_unified_model.pkl cdkapp/lambda/
    cp enhanced_runtime_model.pkl cdkapp/lambda/
    
    cd cdkapp
    npx cdk deploy PredictionLambdaStack --app "python prediction_app.py" --require-approval never
    
    echo "Deployment completed!"
else
    echo "Validation failed, restoring backup models..."
    cp enhanced_unified_model_backup.pkl enhanced_unified_model.pkl
    cp enhanced_runtime_model_backup.pkl enhanced_runtime_model.pkl
fi
```

### 7. Monitoring Retraining Results
```python
# monitor_model_performance.py
import json
from datetime import datetime, timedelta

def compare_predictions_vs_actuals():
    """Compare model predictions against actual values"""
    # Load actual data for recent period
    recent_actuals = load_recent_actuals(days=30)
    
    # Load models and make predictions
    models = load_models()
    
    accuracies = []
    for date, cycle, actual in recent_actuals:
        predicted = make_prediction(models, date, cycle)
        
        if actual['volume'] and predicted['volume']:
            volume_error = abs(predicted['volume'] - actual['volume']) / actual['volume']
            accuracies.append(1 - volume_error)
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    print(f"Average prediction accuracy: {avg_accuracy:.2%}")
    
    return avg_accuracy

if __name__ == "__main__":
    accuracy = compare_predictions_vs_actuals()
    
    # Alert if accuracy drops below threshold
    if accuracy < 0.80:  # 80% accuracy threshold
        print("WARNING: Model accuracy below threshold, consider retraining")
```

## Retraining Schedule

### Recommended Frequency
- **Weekly**: Quick validation of current model performance
- **Monthly**: Full retraining with new data
- **Quarterly**: Comprehensive model review and architecture updates
- **Ad-hoc**: When prediction accuracy drops below 80%

### Data Requirements
- **Minimum**: 30 days of new file_audit.json data
- **Optimal**: 90+ days for seasonal pattern detection
- **Runtime Data**: Corresponding PDE-Runtimes-2025.csv updates

### Quality Checks
1. **Data Completeness**: Verify all required fields present
2. **Data Consistency**: Check for outliers and anomalies
3. **Model Performance**: Validate against holdout test set
4. **Prediction Sanity**: Ensure predictions within reasonable ranges

This retraining guide provides the complete workflow for updating your ML models with new data and deploying them to production.