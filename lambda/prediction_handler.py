import json
import os
import sys
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from prediction_core import get_predictions_api, load_models, load_actuals
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    from simple_prediction import get_simple_predictions

def lambda_handler(event, context):
    """
    Lambda handler for prediction API
    
    Expected input:
    - GET /predict?date=2025-01-15&cycle=Cycle1
    - POST /predict with JSON body: {"date": "2025-01-15", "cycle": "Cycle1"}
    """
    
    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': ''
        }
    
    try:
        # Debug logging
        print(f"Event: {json.dumps(event)}")
        
        # Parse input parameters
        if event.get('httpMethod') == 'GET':
            params = event.get('queryStringParameters') or {}
        else:  # POST
            body_str = event.get('body', '{}')
            print(f"Body string: {body_str}")
            if body_str:
                body = json.loads(body_str)
                params = body
            else:
                params = {}
        
        target_date = params.get('date', datetime.now().strftime('%Y-%m-%d'))
        cycle_type = params.get('cycle')  # Optional - if None, returns both cycles
        
        # Get predictions
        if ML_AVAILABLE:
            models = load_models()
            actuals = load_actuals()
            results = get_predictions_api(models, actuals, target_date, cycle_type)
        else:
            results = get_simple_predictions(target_date, cycle_type)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(results, indent=2)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error'
            })
        }