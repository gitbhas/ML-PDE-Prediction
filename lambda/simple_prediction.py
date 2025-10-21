import json
from datetime import datetime

def get_simple_predictions(target_date, cycle_type=None):
    """Simple rule-based predictions without ML models"""
    dt = datetime.strptime(target_date, '%Y-%m-%d')
    day_name = dt.strftime('%A')
    day_of_week = dt.weekday()
    
    # Skip Saturday PM and Sunday
    cycles = []
    if cycle_type:
        if not (dt.weekday() == 5 and cycle_type == "Cycle3") and dt.weekday() != 6:
            cycles = [cycle_type]
    else:
        if dt.weekday() != 6:  # Not Sunday
            cycles = ['Cycle1']
            if dt.weekday() != 5:  # Not Saturday
                cycles.append('Cycle3')
    
    results = []
    for cycle in cycles:
        # Simple business rules based on day and cycle
        if cycle == 'Cycle1':
            if day_of_week == 0:  # Monday
                file_count = 8
                volume = 17500000
                runtime = 499
            elif day_of_week in [1, 2, 3, 4]:  # Tue-Fri
                file_count = 5
                volume = 12000000
                runtime = 350
            else:  # Saturday
                file_count = 3
                volume = 8000000
                runtime = 200
        else:  # Cycle3
            if day_of_week == 0:  # Monday
                file_count = 1
                volume = 400000
                runtime = 55
            else:  # Tue-Fri
                file_count = 2
                volume = 800000
                runtime = 80
        
        result = {
            'date': target_date,
            'day_name': day_name,
            'cycle': cycle,
            'predictions': {
                'file_count': file_count,
                'volume': volume,
                'runtime_minutes': round(runtime, 1)
            },
            'actuals': {
                'file_count': None,
                'volume': None,
                'runtime_minutes': None
            }
        }
        results.append(result)
    
    return {
        'date': target_date,
        'day_name': day_name,
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'note': 'Using business rules (ML models unavailable)'
    }