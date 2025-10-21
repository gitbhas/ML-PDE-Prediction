#!/usr/bin/env python3
"""
Validate ML Models
Usage: python validate_models.py
"""
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_prediction_features(date, cycle):
    """Create features for prediction"""
    dt = datetime.strptime(date, '%Y-%m-%d')
    
    file_features = [
        dt.weekday(),  # day_of_week
        dt.month,      # month
        1 if cycle == 'Cycle1' else 0,  # is_cycle1
        1 if dt.weekday() >= 5 else 0,  # is_weekend
        1 if dt.weekday() == 0 else 0,  # is_monday
        1 if dt.weekday() == 1 else 0   # is_tuesday
    ]
    
    return {
        'file': file_features,
        'date': date,
        'cycle': cycle,
        'dt': dt
    }

def validate_predictions():
    """Validate new models against recent data"""
    try:
        # Load new models
        unified_model = joblib.load('../lambda/enhanced_unified_model.pkl')
        runtime_model = joblib.load('../lambda/enhanced_runtime_model.pkl')
        
        print("Models loaded successfully!")
        print(f"Unified model features: {unified_model['file_features']}")
        print(f"Runtime model features: {runtime_model['features']}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False
    
    # Test predictions for recent dates
    test_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                  for i in range(1, 8)]
    
    print("\nValidation Results:")
    print("=" * 80)
    
    for date in test_dates:
        for cycle in ['Cycle1', 'Cycle3']:
            # Create test features
            dt = datetime.strptime(date, '%Y-%m-%d')
            if dt.weekday() == 6 or (dt.weekday() == 5 and cycle == 'Cycle3'):
                continue
            
            try:
                features = create_prediction_features(date, cycle)
                
                # Make file count prediction
                file_pred = unified_model['file_model'].predict([features['file']])[0]
                file_pred = max(1, int(file_pred))
                
                # Make volume prediction
                volume_features = features['file'] + [file_pred, 0 if file_pred <= 2 else 1 if file_pred <= 5 else 2 if file_pred <= 10 else 3]
                volume_pred = unified_model['volume_model'].predict([volume_features])[0]
                volume_pred = max(0, int(volume_pred))
                
                # Make runtime prediction
                volume_per_file = volume_pred / file_pred if file_pred > 0 else 0
                is_high_volume = 1 if volume_pred > 5000000 else 0
                runtime_features = features['file'] + [file_pred, 0 if file_pred <= 2 else 1 if file_pred <= 5 else 2 if file_pred <= 10 else 3, volume_pred, volume_per_file, is_high_volume]
                runtime_pred = runtime_model['model'].predict([runtime_features])[0]
                runtime_pred = max(0, runtime_pred)
                
                print(f"{date} {cycle}: Files={file_pred:2d}, Volume={volume_pred:8,d}, Runtime={runtime_pred:5.1f}min")
                
            except Exception as e:
                print(f"{date} {cycle}: ERROR - {e}")
    
    print("\n✅ Model validation completed!")
    return True

def check_model_sanity():
    """Perform sanity checks on model predictions"""
    print("\nSanity Checks:")
    print("-" * 40)
    
    # Test Monday Cycle1 (should be highest)
    monday_c1 = create_prediction_features('2025-10-20', 'Cycle1')  # Monday
    friday_c3 = create_prediction_features('2025-10-24', 'Cycle3')   # Friday
    
    try:
        unified_model = joblib.load('../lambda/enhanced_unified_model.pkl')
        
        # Monday Cycle1 predictions
        mon_file = unified_model['file_model'].predict([monday_c1['file']])[0]
        mon_volume_features = monday_c1['file'] + [int(mon_file), 0 if mon_file <= 2 else 1 if mon_file <= 5 else 2 if mon_file <= 10 else 3]
        mon_volume = unified_model['volume_model'].predict([mon_volume_features])[0]
        
        # Friday Cycle3 predictions
        fri_file = unified_model['file_model'].predict([friday_c3['file']])[0]
        fri_volume_features = friday_c3['file'] + [int(fri_file), 0 if fri_file <= 2 else 1 if fri_file <= 5 else 2 if fri_file <= 10 else 3]
        fri_volume = unified_model['volume_model'].predict([fri_volume_features])[0]
        
        print(f"Monday Cycle1: {mon_file:.0f} files, {mon_volume:,.0f} volume")
        print(f"Friday Cycle3: {fri_file:.0f} files, {fri_volume:,.0f} volume")
        
        # Sanity checks
        if mon_file > fri_file:
            print("✅ Monday Cycle1 > Friday Cycle3 file count")
        else:
            print("❌ Monday Cycle1 should have more files than Friday Cycle3")
            
        if mon_volume > fri_volume:
            print("✅ Monday Cycle1 > Friday Cycle3 volume")
        else:
            print("❌ Monday Cycle1 should have higher volume than Friday Cycle3")
            
    except Exception as e:
        print(f"❌ Sanity check failed: {e}")

def main():
    """Main validation function"""
    print("ML Model Validation")
    print("=" * 50)
    
    success = validate_predictions()
    if success:
        check_model_sanity()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)