#!/usr/bin/env python3
"""
Deploy Prediction API Lambda Function
Usage: python deploy_prediction_api.py
"""
import subprocess
import sys
import os

def run_command(command, cwd=None):
    """Run shell command and return result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Command failed: {e}")
        return False

def main():
    print("Deploying ML Prediction API...")
    
    # Change to CDK directory
    cdk_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Bootstrap CDK (if needed)
    print("\n1. Bootstrapping CDK...")
    run_command("npx cdk bootstrap", cwd=cdk_dir)
    
    # Deploy the stack
    print("\n2. Deploying Prediction Lambda Stack...")
    if not run_command("npx cdk deploy PredictionLambdaStack --app \"python prediction_app.py\" --require-approval never", cdk_dir):
        return False
    
    print("\n✅ Deployment completed successfully!")
    print("\nAPI Usage Examples:")
    print("GET  /predict?date=2025-01-15&cycle=Cycle1")
    print("POST /predict with JSON: {'date': '2025-01-15', 'cycle': 'Cycle1'}")
    
if __name__ == "__main__":
    main()