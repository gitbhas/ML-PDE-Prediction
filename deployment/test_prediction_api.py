#!/usr/bin/env python3
"""
Test the Prediction API locally and remotely
Usage: python test_prediction_api.py [API_URL]
"""
import requests
import json
import sys
from datetime import datetime

def test_local():
    """Test the Lambda function locally"""
    print("Testing Lambda function locally...")
    
    # Import the handler
    sys.path.append('../lambda')
    from prediction_handler import lambda_handler
    
    # Test GET-style event
    get_event = {
        'httpMethod': 'GET',
        'queryStringParameters': {
            'date': '2025-01-15',
            'cycle': 'Cycle1'
        }
    }
    
    # Test POST-style event
    post_event = {
        'httpMethod': 'POST',
        'body': json.dumps({
            'date': '2025-01-15'
        })
    }
    
    print("\n--- GET Request Test ---")
    result = lambda_handler(get_event, {})
    print(f"Status: {result['statusCode']}")
    print(f"Response: {json.loads(result['body'])}")
    
    print("\n--- POST Request Test ---")
    result = lambda_handler(post_event, {})
    print(f"Status: {result['statusCode']}")
    print(f"Response: {json.loads(result['body'])}")

def test_remote(api_url):
    """Test the deployed API"""
    print(f"Testing deployed API at: {api_url}")
    
    # Test GET request
    print("\n--- GET Request Test ---")
    try:
        response = requests.get(f"{api_url}?date=2025-01-15&cycle=Cycle1", timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"GET request failed: {e}")
    
    # Test POST request
    print("\n--- POST Request Test ---")
    try:
        payload = {'date': '2025-01-15'}
        response = requests.post(api_url, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"POST request failed: {e}")

def main():
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
        test_remote(api_url)
    else:
        test_local()

if __name__ == "__main__":
    main()