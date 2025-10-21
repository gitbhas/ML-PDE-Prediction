import pandas as pd
import numpy as np
import json
from datetime import datetime
import pickle

try:
    import joblib
except ImportError:
    joblib = None

def load_models():
    """Load ML models from files"""
    models = {}
    
    def load_model(filename):
        try:
            if joblib:
                return joblib.load(filename)
            else:
                with open(filename, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return None
    
    models['unified'] = load_model('enhanced_unified_model.pkl') or load_model('unified_file_volume_model.pkl')
    models['runtime'] = load_model('enhanced_runtime_model.pkl') or load_model('best_runtime_model_lasso_regression.pkl')
    
    print(f"Models loaded: unified={models['unified'] is not None}, runtime={models['runtime'] is not None}")
    if models['unified']:
        print(f"Unified model type: {type(models['unified'])}")
    if models['runtime']:
        print(f"Runtime model type: {type(models['runtime'])}")
    
    return models

def load_actuals():
    """Load actual data from file_audit.json and PDE-Runtimes-2025.csv"""
    actuals = {}
    
    # Load file count and volume from JSON
    try:
        with open('file_audit.json', 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    date_str = record.get('fil_creatn_dt', '')[:10]
                    if date_str:
                        if date_str not in actuals:
                            actuals[date_str] = {'Cycle1': {'file_count': 0, 'volume': 0, 'runtime': None}, 
                                               'Cycle3': {'file_count': 0, 'volume': 0, 'runtime': None}}
                        
                        time_str = str(record.get('proc_strt_time', '0'))
                        try:
                            hour = int(time_str[:2]) if len(time_str) >= 2 else 12
                        except:
                            hour = 12
                        
                        cycle = 'Cycle1' if hour < 12 else 'Cycle3'
                        actuals[date_str][cycle]['file_count'] += 1
                        actuals[date_str][cycle]['volume'] += int(record.get('in_tot_rec_cnt', 0))
                except:
                    continue
    except:
        pass
    
    # Load runtime data from CSV
    try:
        df = pd.read_csv('PDE-Runtimes-2025.csv')
        for _, row in df.iterrows():
            try:
                date_str = pd.to_datetime(row['PDE date']).strftime('%Y-%m-%d')
                cycle = row['Cycle Type']
                runtime = float(row['Total Runtime'])
                
                if date_str not in actuals:
                    actuals[date_str] = {'Cycle1': {'file_count': 0, 'volume': 0, 'runtime': None}, 
                                       'Cycle3': {'file_count': 0, 'volume': 0, 'runtime': None}}
                
                actuals[date_str][cycle]['runtime'] = runtime
            except:
                continue
    except:
        pass
    
    return actuals

def predict_file_count(models, target_date, cycle_type):
    """Predict file count for given date and cycle"""
    if not models or 'unified' not in models or not models['unified']:
        return 3  # Default fallback
    
    model_data = models['unified']
    dt = datetime.strptime(target_date, '%Y-%m-%d')
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_cycle1 = 1 if cycle_type == "Cycle1" else 0
    is_monday = 1 if day_of_week == 0 else 0
    is_tuesday = 1 if day_of_week == 1 else 0
    
    try:
        if isinstance(model_data, dict) and 'file_features' in model_data:
            features = [day_of_week, month, is_cycle1, is_weekend, is_monday, is_tuesday]
            X = pd.DataFrame([features], columns=model_data['file_features'])
            return int(model_data['file_model'].predict(X)[0])
        else:
            features = {'day_of_week': day_of_week, 'month': month, 'is_am': is_cycle1, 'is_weekend': is_weekend}
            X = pd.DataFrame([features])
            return int(model_data['file_model'].predict(X)[0])
    except:
        return 3  # Default fallback

def predict_volume(models, target_date, cycle_type, file_count=None):
    """Predict volume for given date and cycle"""
    if not models or 'unified' not in models or not models['unified']:
        return 2000000  # Default fallback
    
    model_data = models['unified']
    dt = datetime.strptime(target_date, '%Y-%m-%d')
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_cycle1 = 1 if cycle_type == "Cycle1" else 0
    is_monday = 1 if day_of_week == 0 else 0
    is_tuesday = 1 if day_of_week == 1 else 0
    
    try:
        if isinstance(model_data, dict) and 'volume_features' in model_data and file_count is not None:
            file_count_category = 0 if file_count <= 2 else (1 if file_count <= 5 else (2 if file_count <= 10 else 3))
            features = [day_of_week, month, is_cycle1, is_weekend, is_monday, is_tuesday, file_count, file_count_category]
            X = pd.DataFrame([features], columns=model_data['volume_features'])
            return int(model_data['volume_model'].predict(X)[0])
        else:
            features = {'day_of_week': day_of_week, 'month': month, 'is_am': is_cycle1, 'is_weekend': is_weekend}
            X = pd.DataFrame([features])
            return int(model_data['volume_model'].predict(X)[0])
    except:
        return 2000000  # Default fallback

def predict_runtime(models, total_records, target_date, cycle_type, file_count=None):
    """Predict runtime for given parameters"""
    if not models or 'runtime' not in models or not models['runtime']:
        return 45.0  # Default fallback
    
    model_data = models['runtime']
    dt = datetime.strptime(target_date, '%Y-%m-%d')
    day_of_week = dt.weekday()
    month = dt.month
    is_cycle1 = 1 if cycle_type == "Cycle1" else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    is_monday = 1 if day_of_week == 0 else 0
    is_tuesday = 1 if day_of_week == 1 else 0
    
    if isinstance(model_data, dict) and 'features' in model_data:
        volume_per_file = total_records / file_count if file_count and file_count > 0 else total_records
        is_high_volume = 1 if total_records > 5000000 else 0
        file_count_category = 0 if file_count <= 2 else (1 if file_count <= 5 else (2 if file_count <= 10 else 3)) if file_count else 1
        
        feature_values = [day_of_week, month, is_cycle1, is_weekend, is_monday, is_tuesday, 
                         file_count or 1, file_count_category, total_records, volume_per_file, is_high_volume]
        X = pd.DataFrame([feature_values], columns=model_data['features'])
        return model_data['model'].predict(X)[0]
    
    elif hasattr(model_data, 'predict'):
        X = pd.DataFrame([[total_records, day_of_week, month, is_cycle1]], 
                        columns=['total_records', 'day_of_week', 'month', 'is_cycle1'])
        return model_data.predict(X)[0]
    
    else:
        features = {'total_records': total_records, 'day_of_week': day_of_week, 'month': month,
                   'is_cycle1': is_cycle1, 'is_weekend': is_weekend}
        feature_cols = model_data.get('features', list(features.keys()))
        X = pd.DataFrame([[features.get(col, 0) for col in feature_cols]], columns=feature_cols)
        return model_data['model'].predict(X)[0]

def get_predictions_api(models, actuals, target_date, cycle_type=None):
    """Get predictions for API response"""
    dt = datetime.strptime(target_date, '%Y-%m-%d')
    day_name = dt.strftime('%A')
    
    # Determine cycles to process
    if cycle_type:
        cycles = [cycle_type]
    else:
        cycles = ['Cycle1', 'Cycle3']
    
    results = []
    
    for cycle in cycles:
        # Skip Saturday PM and Sunday
        if (dt.weekday() == 5 and cycle == "Cycle3") or dt.weekday() == 6:
            continue
        
        # Get predictions
        pred_file_count = predict_file_count(models, target_date, cycle)
        pred_file_count = max(1, pred_file_count) if pred_file_count is not None else 1
        pred_total_volume = predict_volume(models, target_date, cycle, pred_file_count)
        pred_runtime = predict_runtime(models, pred_total_volume, target_date, cycle, pred_file_count) if pred_total_volume else None
        
        # Get actuals
        actual_data = actuals.get(target_date, {}).get(cycle, {})
        
        result = {
            'date': target_date,
            'day_name': day_name,
            'cycle': cycle,
            'predictions': {
                'file_count': pred_file_count,
                'volume': pred_total_volume,
                'runtime_minutes': round(pred_runtime, 1) if pred_runtime else None
            },
            'actuals': {
                'file_count': actual_data.get('file_count'),
                'volume': actual_data.get('volume'),
                'runtime_minutes': actual_data.get('runtime')
            }
        }
        
        results.append(result)
    
    return {
        'date': target_date,
        'day_name': day_name,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }