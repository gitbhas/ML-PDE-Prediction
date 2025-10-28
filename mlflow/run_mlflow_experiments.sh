#!/bin/bash
# Run MLflow Model Experiments
# Usage: ./run_mlflow_experiments.sh

echo "Setting up MLflow experiments..."

# Install MLflow if not already installed
pip install --upgrade setuptools pip
pip install mlflow pandas numpy scikit-learn joblib matplotlib

# Run experiments
echo "Running model experiments..."
python mlflow_experiments.py

# Start MLflow UI
echo ""
echo "Starting MLflow UI..."
echo "Access at: http://localhost:5000"
echo "Press Ctrl+C to stop"

mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000