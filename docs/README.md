# ML Prediction API - Project Structure

## Directory Organization

```
cdkapp/
├── deployment/           # Deployment scripts and tools
│   ├── deploy_prediction_api.py    # Main deployment script
│   └── test_prediction_api.py      # API testing script
├── training/            # Model training and validation
│   ├── proper_ml_training.py       # Complete training pipeline
│   ├── validate_models.py          # Model validation script
│   └── retrain_pipeline.sh         # Automated retraining pipeline
├── monitoring/          # Performance monitoring
│   └── monitor_model_performance.py # Accuracy tracking and alerts
├── docs/               # Documentation
│   └── README.md       # This file
├── web/                # Web interface files
│   ├── index.html      # Web UI
│   └── script.js       # Frontend JavaScript
├── lambda/             # Lambda function code and models
│   ├── prediction_handler.py       # Lambda entry point
│   ├── prediction_core.py          # ML prediction logic
│   ├── simple_prediction.py        # Fallback predictions
│   ├── Dockerfile      # Container definition
│   ├── requirements.txt # Python dependencies
│   └── *.pkl          # ML model files
└── stacks/             # CDK infrastructure
    ├── prediction_lambda_stack.py  # Lambda + API Gateway
    └── static_web_stack.py         # S3 + CloudFront (optional)
```

## Quick Start

### 1. Deploy the API
```bash
cd deployment
python deploy_prediction_api.py
```

### 2. Test the API
```bash
cd deployment
python test_prediction_api.py https://your-api-url/prod/predict
```

### 3. Retrain Models
```bash
cd training
python proper_ml_training.py
python validate_models.py
```

### 4. Monitor Performance
```bash
cd monitoring
python monitor_model_performance.py https://your-api-url/prod/predict
```

## Usage Examples

### API Calls
```bash
# GET request
curl "https://your-api-url/prod/predict?date=2025-10-20&cycle=Cycle1"

# POST request
curl -X POST https://your-api-url/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2025-10-20", "cycle": "Cycle1"}'
```

### Web Interface
Open `web/index.html` in your browser to use the interactive web interface.

## Model Files

### Primary Models
- `enhanced_unified_model.pkl` - File count and volume predictions
- `enhanced_runtime_model.pkl` - Runtime predictions

### Fallback Models
- `unified_file_volume_model.pkl` - Simplified unified model
- `best_runtime_model_lasso_regression.pkl` - Lasso regression runtime model

## Data Files
- `file_audit.json` - Production file processing logs
- `PDE-Runtimes-2025.csv` - Runtime measurements

## Maintenance Schedule

### Daily
- Monitor API health and response times
- Check CloudWatch logs for errors

### Weekly
- Run performance monitoring script
- Review prediction accuracy trends

### Monthly
- Retrain models with new data
- Update documentation if needed

### Quarterly
- Comprehensive model architecture review
- Performance optimization analysis

## Troubleshooting

### Common Issues

1. **API Timeout**
   - Increase Lambda timeout in CDK stack
   - Check model loading performance

2. **CORS Errors**
   - Verify CORS headers in Lambda response
   - Check API Gateway CORS configuration

3. **Model Loading Failures**
   - Ensure all .pkl files are in lambda/ directory
   - Check scikit-learn version compatibility

4. **Poor Prediction Accuracy**
   - Run monitoring script to identify issues
   - Retrain models with recent data
   - Check for data quality issues

### Support
- Check CloudWatch logs for detailed error messages
- Use test scripts to isolate issues
- Review model validation results

## Security Notes

- S3 bucket is private with CloudFront OAI access
- Lambda function has minimal IAM permissions
- API Gateway has CORS enabled for web access
- No sensitive data stored in container image

## Performance Considerations

- Container cold start: ~2-3 seconds
- Model loading: ~1-2 seconds
- Prediction inference: <100ms
- Consider provisioned concurrency for production

This structure provides a complete, maintainable ML prediction system with proper separation of concerns and comprehensive tooling.