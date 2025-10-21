#!/usr/bin/env python3
"""
Monitor ML Model Performance
Usage: python monitor_model_performance.py
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def load_recent_actuals(days=30):
    """Load actual data for recent period"""
    actuals = []
    
    try:
        # Load file audit data
        file_path = '../file_audit.json'
        if not os.path.exists(file_path):
            file_path = '../../file_audit.json'
            
        daily_stats = {}
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    date_str = record.get('fil_creatn_dt', '')[:10]
                    if not date_str:
                        continue
                    
                    # Only process recent dates
                    record_date = datetime.strptime(date_str, '%Y-%m-%d')
                    cutoff_date = datetime.now() - timedelta(days=days)
                    if record_date < cutoff_date:
                        continue
                    
                    # Determine cycle
                    time_str = str(record.get('proc_strt_time', '0'))
                    hour = int(time_str[:2]) if len(time_str) >= 2 else 12
                    cycle = 'Cycle1' if hour < 12 else 'Cycle3'
                    
                    key = (date_str, cycle)
                    if key not in daily_stats:
                        daily_stats[key] = {'file_count': 0, 'volume': 0}
                    
                    daily_stats[key]['file_count'] += 1
                    daily_stats[key]['volume'] += int(record.get('in_tot_rec_cnt', 0))
                    
                except:
                    continue
        
        # Load runtime data
        runtime_path = '../PDE-Runtimes-2025.csv'
        if not os.path.exists(runtime_path):
            runtime_path = '../../PDE-Runtimes-2025.csv'
            
        runtime_df = pd.read_csv(runtime_path)
        
        # Combine data
        for (date_str, cycle), stats in daily_stats.items():
            # Find matching runtime
            runtime_match = runtime_df[
                (pd.to_datetime(runtime_df['PDE date']).dt.strftime('%Y-%m-%d') == date_str) &
                (runtime_df['Cycle Type'] == cycle)
            ]
            
            runtime = float(runtime_match.iloc[0]['Total Runtime']) if not runtime_match.empty else None
            
            actuals.append({
                'date': date_str,
                'cycle': cycle,
                'file_count': stats['file_count'],
                'volume': stats['volume'],
                'runtime': runtime
            })
    
    except Exception as e:
        print(f"Error loading actuals: {e}")
    
    return actuals

def make_api_prediction(api_url, date, cycle):
    """Make prediction using deployed API"""
    try:
        payload = {'date': date, 'cycle': cycle}
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                result = data['results'][0]
                return result['predictions']
        
    except Exception as e:
        print(f"API prediction failed for {date} {cycle}: {e}")
    
    return None

def calculate_accuracy(predicted, actual):
    """Calculate prediction accuracy"""
    if predicted is None or actual is None or actual == 0:
        return None
    
    error = abs(predicted - actual) / actual
    accuracy = max(0, 1 - error)
    return accuracy

def compare_predictions_vs_actuals(api_url=None, days=30):
    """Compare model predictions against actual values"""
    print(f"Analyzing prediction accuracy for last {days} days...")
    print("=" * 70)
    
    # Load actual data
    recent_actuals = load_recent_actuals(days)
    
    if not recent_actuals:
        print("No recent actual data found!")
        return 0
    
    print(f"Found {len(recent_actuals)} actual data points")
    
    accuracies = {
        'file_count': [],
        'volume': [],
        'runtime': []
    }
    
    detailed_results = []
    
    for actual in recent_actuals:
        if api_url:
            # Use API for predictions
            predicted = make_api_prediction(api_url, actual['date'], actual['cycle'])
        else:
            # Use local models (if available)
            predicted = None  # Would need to implement local prediction
        
        if predicted:
            # Calculate accuracies
            file_acc = calculate_accuracy(predicted.get('file_count'), actual['file_count'])
            volume_acc = calculate_accuracy(predicted.get('volume'), actual['volume'])
            runtime_acc = calculate_accuracy(predicted.get('runtime_minutes'), actual['runtime'])
            
            if file_acc is not None:
                accuracies['file_count'].append(file_acc)
            if volume_acc is not None:
                accuracies['volume'].append(volume_acc)
            if runtime_acc is not None:
                accuracies['runtime'].append(runtime_acc)
            
            detailed_results.append({
                'date': actual['date'],
                'cycle': actual['cycle'],
                'predicted': predicted,
                'actual': actual,
                'accuracies': {
                    'file_count': file_acc,
                    'volume': volume_acc,
                    'runtime': runtime_acc
                }
            })
    
    # Calculate average accuracies
    avg_accuracies = {}
    for metric, acc_list in accuracies.items():
        if acc_list:
            avg_accuracies[metric] = sum(acc_list) / len(acc_list)
        else:
            avg_accuracies[metric] = 0
    
    # Display results
    print("\nPrediction Accuracy Summary:")
    print("-" * 40)
    for metric, accuracy in avg_accuracies.items():
        status = "✅" if accuracy >= 0.80 else "⚠️" if accuracy >= 0.70 else "❌"
        print(f"{metric:12}: {accuracy:.1%} {status}")
    
    overall_accuracy = sum(avg_accuracies.values()) / len(avg_accuracies) if avg_accuracies else 0
    print(f"{'Overall':12}: {overall_accuracy:.1%}")
    
    # Show detailed results for recent dates
    print(f"\nDetailed Results (Last 10 entries):")
    print("-" * 70)
    print(f"{'Date':<12} {'Cycle':<8} {'Metric':<12} {'Predicted':<12} {'Actual':<12} {'Accuracy':<10}")
    print("-" * 70)
    
    for result in detailed_results[-10:]:
        date = result['date']
        cycle = result['cycle']
        pred = result['predicted']
        actual = result['actual']
        acc = result['accuracies']
        
        # File count
        if acc['file_count'] is not None:
            print(f"{date:<12} {cycle:<8} {'Files':<12} {pred.get('file_count', 'N/A'):<12} {actual['file_count']:<12} {acc['file_count']:.1%}")
        
        # Volume
        if acc['volume'] is not None:
            print(f"{'':<12} {'':<8} {'Volume':<12} {pred.get('volume', 'N/A'):<12} {actual['volume']:<12} {acc['volume']:.1%}")
        
        # Runtime
        if acc['runtime'] is not None:
            print(f"{'':<12} {'':<8} {'Runtime':<12} {pred.get('runtime_minutes', 'N/A'):<12} {actual['runtime']:<12} {acc['runtime']:.1%}")
        
        print()
    
    # Alert if accuracy drops below threshold
    if overall_accuracy < 0.80:
        print("\n🚨 WARNING: Overall model accuracy below 80% threshold!")
        print("   Consider retraining models with recent data")
    elif overall_accuracy < 0.90:
        print("\n⚠️  NOTICE: Model accuracy could be improved")
        print("   Monitor trends and consider retraining if accuracy continues to decline")
    else:
        print("\n✅ Model performance is good!")
    
    return overall_accuracy

def trend_analysis(days=90):
    """Analyze prediction accuracy trends over time"""
    print(f"\nTrend Analysis (Last {days} days):")
    print("-" * 40)
    
    # This would require storing historical accuracy data
    # For now, just show a placeholder
    print("Trend analysis requires historical accuracy tracking")
    print("Consider implementing a database to store daily accuracy metrics")

def main():
    """Main monitoring function"""
    print("ML Model Performance Monitor")
    print("=" * 50)
    
    # Check if API URL provided
    api_url = None
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
        print(f"Using API: {api_url}")
    else:
        print("No API URL provided, using local analysis")
    
    # Run accuracy analysis
    accuracy = compare_predictions_vs_actuals(api_url, days=30)
    
    # Run trend analysis
    trend_analysis()
    
    print(f"\nMonitoring completed. Overall accuracy: {accuracy:.1%}")
    
    return accuracy >= 0.80

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)