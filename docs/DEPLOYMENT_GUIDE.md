# ML Prediction API Deployment Guide

## Overview
This guide explains how the `prediction_app.py` deployment works, including Lambda containerization and ECR integration.

## Architecture Components

### 1. CDK Stack (`prediction_lambda_stack.py`)
```python
prediction_lambda = _lambda.Function(
    self, 'PredictionFunction',
    runtime=_lambda.Runtime.FROM_IMAGE,
    handler=_lambda.Handler.FROM_IMAGE,
    code=_lambda.Code.from_asset_image('./lambda'),
    timeout=Duration.seconds(300),
    memory_size=2048
)
```

**Key Points:**
- `FROM_IMAGE` runtime tells Lambda to use container image
- `from_asset_image('./lambda')` points to Dockerfile directory
- CDK automatically handles ECR push during deployment

### 2. Container Image (`lambda/Dockerfile`)
```dockerfile
FROM public.ecr.aws/lambda/python:3.12

# Install dependencies with SSL workaround
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -r requirements.txt

# Copy function code and ML models
COPY . ${LAMBDA_TASK_ROOT}

CMD ["prediction_handler.lambda_handler"]
```

**What happens:**
- Base image: AWS Lambda Python 3.12 runtime
- Installs: pandas, numpy, scikit-learn, joblib
- Copies: All .pkl model files, data files, Python code
- Entry point: `prediction_handler.lambda_handler`

## Deployment Process

### Step 1: CDK Preparation
```bash
cd cdkapp
npx cdk deploy PredictionLambdaStack --app "python prediction_app.py" --require-approval never
```

### Step 2: Docker Build Process
CDK automatically:
1. **Builds Docker Image**
   ```bash
   docker build --tag cdkasset-[hash] ./lambda
   ```

2. **Creates ECR Repository**
   - Repository name: `cdk-[hash]-container-assets-[account]-[region]`
   - Lifecycle policy: Delete untagged images after 1 day

3. **Pushes to ECR**
   ```bash
   docker tag cdkasset-[hash] [account].dkr.ecr.[region].amazonaws.com/cdk-[hash]:latest
   docker push [account].dkr.ecr.[region].amazonaws.com/cdk-[hash]:latest
   ```

### Step 3: Lambda Function Creation
CDK creates Lambda function with:
- **Image URI**: Points to ECR repository
- **Configuration**: 5min timeout, 2GB memory
- **Environment**: PYTHONPATH set to /var/task

### Step 4: API Gateway Integration
- **REST API**: Created with CORS enabled
- **Methods**: GET, POST on `/predict` resource
- **Integration**: Lambda proxy integration

## Container vs Layer Comparison

### Why Container Over Layers?

| Aspect | Layers (250MB limit) | Container (10GB limit) |
|--------|---------------------|------------------------|
| **Size Limit** | 250MB total | 10GB |
| **ML Libraries** | ❌ Too large | ✅ Fits easily |
| **Model Files** | ❌ Limited space | ✅ All models included |
| **Dependencies** | Complex management | Simple Dockerfile |
| **Cold Start** | Faster | Slightly slower |

### Container Benefits
- **No Size Constraints**: All ML libraries + models fit
- **Simplified Dependencies**: Single Dockerfile manages everything
- **Reproducible Builds**: Same environment locally and in AWS
- **Version Control**: Docker image tags for versioning

## File Structure
```
cdkapp/
├── lambda/
│   ├── Dockerfile                 # Container definition
│   ├── requirements.txt           # Python dependencies
│   ├── prediction_handler.py      # Lambda entry point
│   ├── prediction_core.py         # ML prediction logic
│   ├── *.pkl                     # ML model files
│   ├── file_audit.json           # Training data
│   └── PDE-Runtimes-2025.csv     # Runtime data
├── stacks/
│   └── prediction_lambda_stack.py # CDK infrastructure
└── prediction_app.py             # CDK app entry point
```

## ECR Repository Management

### Automatic ECR Operations
CDK handles:
- Repository creation with unique names
- Image lifecycle policies
- Authentication and push
- Lambda function updates

### Manual ECR Commands (if needed)
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [account].dkr.ecr.us-east-1.amazonaws.com

# List repositories
aws ecr describe-repositories

# List images in repository
aws ecr list-images --repository-name cdk-[hash]-container-assets-[account]-us-east-1
```

## Deployment Troubleshooting

### Common Issues

1. **Docker Not Running**
   ```
   ERROR: error during connect: Head "http://...": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified
   ```
   **Solution**: Start Docker Desktop

2. **SSL Certificate Issues**
   ```
   ERROR: No matching distribution found for pandas==2.0.3
   ```
   **Solution**: Added `--trusted-host` flags in Dockerfile

3. **Layer Size Exceeded**
   ```
   Layers consume more than the available size of 262144000 bytes
   ```
   **Solution**: Switched to container image approach

4. **CORS Errors**
   ```
   Access to fetch ... has been blocked by CORS policy
   ```
   **Solution**: Added proper CORS headers in Lambda response

### Performance Optimization

**Cold Start Optimization:**
- Container images have ~2-3 second cold start
- Consider provisioned concurrency for production
- Optimize Dockerfile layers for caching

**Memory Configuration:**
- 2GB memory for ML model loading
- Adjust based on actual usage patterns
- Monitor CloudWatch metrics

## Monitoring and Logs

### CloudWatch Integration
- **Function Logs**: `/aws/lambda/PredictionLambdaStack-PredictionFunction[hash]`
- **Metrics**: Duration, memory usage, error rates
- **X-Ray Tracing**: Enabled by default

### Debug Commands
```bash
# View recent logs
aws logs tail /aws/lambda/PredictionLambdaStack-PredictionFunction[hash] --follow

# Test function directly
aws lambda invoke --function-name PredictionLambdaStack-PredictionFunction[hash] --payload '{"httpMethod":"POST","body":"{\"date\":\"2025-10-20\"}"}' response.json
```

## Cost Considerations

### Container vs Layer Costs
- **Storage**: ECR charges ~$0.10/GB/month
- **Compute**: Same Lambda pricing model
- **Data Transfer**: Minimal for image pulls

### Optimization Tips
- Use multi-stage builds to reduce image size
- Leverage Docker layer caching
- Clean up unused ECR images regularly

## Security Best Practices

### Container Security
- Base image from AWS official registry
- No secrets in Dockerfile or image
- Regular base image updates
- Minimal attack surface

### IAM Permissions
CDK automatically creates:
- Lambda execution role
- ECR repository policies
- API Gateway permissions

## Production Considerations

### Scaling
- **Concurrent Executions**: Default 1000, can increase
- **Provisioned Concurrency**: For consistent performance
- **API Gateway Throttling**: 10,000 requests/second default

### CI/CD Integration
```yaml
# Example GitHub Actions
- name: Deploy Lambda
  run: |
    cd cdkapp
    npx cdk deploy PredictionLambdaStack --require-approval never
```

### Environment Management
- Use CDK context for different environments
- Separate stacks for dev/staging/prod
- Environment-specific configuration

## Summary

The containerized Lambda approach provides:
- **Scalability**: Handle ML workloads without size limits
- **Simplicity**: Single deployment command
- **Reliability**: Consistent environment across deployments
- **Maintainability**: Standard Docker workflows

This architecture successfully converts the original `predict_all.py` script into a serverless, scalable API while maintaining all ML functionality and data processing capabilities.