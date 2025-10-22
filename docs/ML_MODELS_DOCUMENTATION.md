# ML Models Documentation

## Overview
The prediction API uses multiple machine learning models to predict file counts, data volumes, and processing runtimes for PDE (Parallel Data Engine) cycles.

## Model Architecture

### Primary Models Used

#### 1. Enhanced Unified Model (`enhanced_unified_model.pkl`)
**Purpose**: Predicts both file counts and data volumes
**Features**:
- `day_of_week` (0-6, Monday=0)
- `month` (1-12)
- `is_cycle1` (1 for Cycle1, 0 for Cycle3)
- `is_weekend` (1 for Sat/Sun, 0 for weekdays)
- `is_monday` (1 for Monday, 0 otherwise)
- `is_tuesday` (1 for Tuesday, 0 otherwise)
- `file_count` (for volume prediction)
- `file_count_category` (0: ≤2, 1: 3-5, 2: 6-10, 3: >10)

**Structure**:
```python
{
    'file_model': sklearn_model,      # Predicts file count
    'volume_model': sklearn_model,    # Predicts data volume
    'file_features': [feature_names], # Feature column names
    'volume_features': [feature_names]
}
```

#### 2. Enhanced Runtime Model (`enhanced_runtime_model.pkl`)
**Purpose**: Predicts processing runtime in minutes
**Features**:
- All temporal features (day_of_week, month, cycle, etc.)
- `file_count` (number of files to process)
- `file_count_category` (categorical file count)
- `total_records` (total data volume)
- `volume_per_file` (average records per file)
- `is_high_volume` (1 if >5M records, 0 otherwise)

**Structure**:
```python
{
    'model': sklearn_model,
    'features': [feature_names]
}
```

### Fallback Models

#### 3. Unified File Volume Model (`unified_file_volume_model.pkl`)
**Purpose**: Backup model with simpler feature set
**Features**: Basic temporal features only

#### 4. Best Runtime Model (`best_runtime_model_lasso_regression.pkl`)
**Purpose**: Lasso regression model for runtime prediction
**Features**: `total_records`, `day_of_week`, `month`, `is_cycle1`

## Training Data Sources

### 1. File Audit Data (`file_audit.json`)
**Source**: Production PDE file processing logs
**Format**: JSONL (JSON Lines)
**Key Fields**:
```json
{
    "fil_creatn_dt": "2025-10-20T08:30:00",  // File creation timestamp
    "proc_strt_time": "0830",                // Processing start time (HHMM)
    "in_tot_rec_cnt": 2500000                // Input record count
}
```

**Processing Logic**:
- Extract date from `fil_creatn_dt` (YYYY-MM-DD format)
- Determine cycle from `proc_strt_time` (hour < 12 = Cycle1, else Cycle3)
- Aggregate by date/cycle: count files, sum volumes

### 2. Runtime Data (`PDE-Runtimes-2025.csv`)
**Source**: Production runtime measurements
**Format**: CSV
**Key Fields**:
```csv
PDE date,Cycle Type,Total Runtime
2025-10-20,Cycle1,498.5
2025-10-20,Cycle3,54.2
```

## Feature Engineering

### Temporal Features
```python
dt = datetime.strptime(target_date, '%Y-%m-%d')
day_of_week = dt.weekday()        # 0=Monday, 6=Sunday
month = dt.month                  # 1-12
is_weekend = 1 if day_of_week >= 5 else 0
is_monday = 1 if day_of_week == 0 else 0
is_tuesday = 1 if day_of_week == 1 else 0
```

### Cycle Features
```python
is_cycle1 = 1 if cycle_type == "Cycle1" else 0
```

### Volume Features
```python
# File count categorization
file_count_category = (
    0 if file_count <= 2 else
    1 if file_count <= 5 else
    2 if file_count <= 10 else 3
)

# Volume analysis
volume_per_file = total_records / file_count if file_count > 0 else total_records
is_high_volume = 1 if total_records > 5000000 else 0
```

## Model Training Process

### 1. Data Preparation
```python
# Load and parse file audit data
with open('file_audit.json', 'r') as f:
    for line in f:
        record = json.loads(line.strip())
        # Extract features and aggregate by date/cycle
```

### 2. Feature Matrix Creation
```python
# Create feature matrix
X = pd.DataFrame({
    'day_of_week': day_features,
    'month': month_features,
    'is_cycle1': cycle_features,
    # ... additional features
})

# Target variables
y_file_count = file_count_targets
y_volume = volume_targets
y_runtime = runtime_targets
```

### 3. Model Training
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# File count model
file_model = RandomForestRegressor(n_estimators=100, random_state=42)
file_model.fit(X_file, y_file_count)

# Volume model (uses file count as feature)
volume_model = RandomForestRegressor(n_estimators=100, random_state=42)
volume_model.fit(X_volume, y_volume)

# Runtime model
runtime_model = LassoCV(cv=5, random_state=42)
runtime_model.fit(X_runtime, y_runtime)
```

### 4. Model Persistence
```python
import joblib

# Save unified model
unified_model = {
    'file_model': file_model,
    'volume_model': volume_model,
    'file_features': file_feature_names,
    'volume_features': volume_feature_names
}
joblib.dump(unified_model, 'enhanced_unified_model.pkl')

# Save runtime model
runtime_model_dict = {
    'model': runtime_model,
    'features': runtime_feature_names
}
joblib.dump(runtime_model_dict, 'enhanced_runtime_model.pkl')
```

## Business Rules and Constraints

### Processing Schedule
- **Monday-Friday**: Both Cycle1 (AM) and Cycle3 (PM)
- **Saturday**: Only Cycle1 (AM)
- **Sunday**: No processing

### Validation Logic
```python
# Skip invalid combinations
if dt.weekday() == 5 and cycle_type == "Cycle3":  # Saturday PM
    return {'skip': True, 'reason': 'No Cycle3 on Saturday'}

if dt.weekday() == 6:  # Sunday
    return {'skip': True, 'reason': 'No processing on Sunday'}
```

### Minimum Constraints
```python
# Ensure minimum file count for processing days
pred_file_count = max(1, pred_file_count) if pred_file_count is not None else 1
```

## Model Performance Characteristics

### Typical Predictions

#### Cycle1 (Morning)
- **Monday**: 8 files, 17.5M records, 499 minutes
- **Tue-Fri**: 5 files, 12M records, 350 minutes  
- **Saturday**: 3 files, 8M records, 200 minutes

#### Cycle3 (Evening)
- **Monday**: 1 file, 400K records, 55 minutes
- **Tue-Fri**: 2 files, 800K records, 80 minutes

### Model Accuracy Factors
- **Seasonal Patterns**: Month-based variations
- **Day-of-Week Effects**: Monday typically highest volume
- **Cycle Dependencies**: Cycle1 > Cycle3 in volume/runtime
- **File Count Correlation**: Higher file counts → higher volumes

## Model Loading in Production

### Loading Hierarchy
```python
def load_models():
    models = {}
    
    # Try enhanced models first
    models['unified'] = (
        load_model('enhanced_unified_model.pkl') or 
        load_model('unified_file_volume_model.pkl')
    )
    
    models['runtime'] = (
        load_model('enhanced_runtime_model.pkl') or 
        load_model('best_runtime_model_lasso_regression.pkl')
    )
    
    return models
```

### Fallback Strategy
1. **Primary**: Enhanced models with full feature sets
2. **Secondary**: Simplified models with basic features
3. **Tertiary**: Business rule-based predictions
4. **Final**: Default static values

## Model Versioning and Updates

### Current Model Versions
- `enhanced_unified_model.pkl` - Latest unified file/volume predictor
- `enhanced_runtime_model.pkl` - Latest runtime predictor with volume features
- `best_runtime_model_lasso_regression.pkl` - Lasso regression baseline

### Update Process
1. **Data Collection**: Accumulate new file_audit.json and runtime data
2. **Retraining**: Run training scripts with updated datasets
3. **Validation**: Compare predictions against recent actuals
4. **Deployment**: Replace .pkl files in Lambda container
5. **Monitoring**: Track prediction accuracy post-deployment

## Prediction Pipeline

### Sequential Prediction Flow
```python
# 1. Predict file count
file_count = predict_file_count(models, date, cycle)

# 2. Predict volume (using file count)
volume = predict_volume(models, date, cycle, file_count)

# 3. Predict runtime (using volume and file count)
runtime = predict_runtime(models, volume, date, cycle, file_count)
```

### Feature Dependencies
- Volume prediction depends on file count prediction
- Runtime prediction depends on both volume and file count
- All predictions use temporal features (date, cycle, day-of-week)

## Model Limitations and Assumptions

### Known Limitations
- **Historical Bias**: Models trained on historical patterns
- **Seasonal Drift**: May not capture new seasonal patterns
- **Outlier Sensitivity**: Extreme values may skew predictions
- **Feature Correlation**: Some features may be correlated

### Assumptions
- Processing patterns remain relatively stable
- File audit data accurately represents processing load
- Runtime measurements are consistent and reliable
- Business rules (schedule, cycles) remain unchanged

## Monitoring and Maintenance

### Key Metrics to Monitor
- **Prediction Accuracy**: Compare predicted vs actual values
- **Model Drift**: Track prediction error trends over time
- **Data Quality**: Monitor input data completeness and consistency
- **Performance**: Model inference time and resource usage

### Maintenance Schedule
- **Weekly**: Review prediction accuracy
- **Monthly**: Analyze model performance trends
- **Quarterly**: Retrain models with new data
- **Annually**: Comprehensive model architecture review

This documentation provides the foundation for understanding, maintaining, and improving the ML prediction models used in the PDE processing system.