# ML Prediction API

Serverless ML prediction system using AWS Lambda, API Gateway, and containerized models.

## Quick Start

```bash
# Deploy API
cd deployment
python deploy_prediction_api.py

# Test API
python test_prediction_api.py https://your-api-url/prod/predict

# Retrain models
cd ../training
python proper_ml_training.py
```

## API Usage

```bash
# GET request
curl "https://your-api-url/prod/predict?date=2025-10-20&cycle=Cycle1"

# POST request
curl -X POST https://your-api-url/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2025-10-20", "cycle": "Cycle1"}'
```

## Web Interface

Open `web/index.html` in your browser for interactive predictions.

## Architecture

- **Lambda Function**: Containerized Python with ML models
- **API Gateway**: RESTful API with CORS
- **Models**: RandomForest + Lasso regression for predictions
- **Data**: File audit logs + runtime measurements

## Directory Structure

```
├── deployment/     # Deployment scripts
├── training/       # Model training
├── monitoring/     # Performance tracking
├── web/           # Web interface
├── lambda/        # Lambda code & models
├── stacks/        # CDK infrastructure
└── docs/          # Documentation
```

## Requirements

- AWS CLI configured
- Docker Desktop running
- Node.js with CDK
- Python 3.12+

## Documentation

See `docs/` folder for detailed guides on deployment, training, and monitoring.