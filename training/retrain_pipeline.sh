#!/bin/bash
# Automated Model Retraining Pipeline
# Usage: ./retrain_pipeline.sh

echo "Starting automated retraining pipeline..."
echo "========================================"

# Step 1: Backup current models
echo "1. Backing up current models..."
if [ -f "../lambda/enhanced_unified_model.pkl" ]; then
    cp ../lambda/enhanced_unified_model.pkl ../lambda/enhanced_unified_model_backup.pkl
    echo "   ✅ Unified model backed up"
fi

if [ -f "../lambda/enhanced_runtime_model.pkl" ]; then
    cp ../lambda/enhanced_runtime_model.pkl ../lambda/enhanced_runtime_model_backup.pkl
    echo "   ✅ Runtime model backed up"
fi

# Step 2: Download latest data (if from S3)
echo "2. Updating training data..."
# Uncomment if data is stored in S3
# aws s3 cp s3://ddps-prd-oes-staging/FILEAU.ML.EXT000.json ../file_audit.json
# aws s3 cp s3://your-bucket/PDE-Runtimes-2025.csv ../PDE-Runtimes-2025.csv
echo "   ℹ️  Using local data files"

# Step 3: Run training
echo "3. Running model training..."
python proper_ml_training.py

if [ $? -ne 0 ]; then
    echo "   ❌ Training failed!"
    exit 1
fi

echo "   ✅ Training completed"

# Step 4: Validate models
echo "4. Validating new models..."
python validate_models.py

if [ $? -ne 0 ]; then
    echo "   ❌ Validation failed, restoring backup models..."
    
    if [ -f "../lambda/enhanced_unified_model_backup.pkl" ]; then
        cp ../lambda/enhanced_unified_model_backup.pkl ../lambda/enhanced_unified_model.pkl
    fi
    
    if [ -f "../lambda/enhanced_runtime_model_backup.pkl" ]; then
        cp ../lambda/enhanced_runtime_model_backup.pkl ../lambda/enhanced_runtime_model.pkl
    fi
    
    echo "   ✅ Backup models restored"
    exit 1
fi

echo "   ✅ Validation passed"

# Step 5: Deploy if validation passes
echo "5. Deploying new models..."
cd ..
npx cdk deploy PredictionLambdaStack --app "python prediction_app.py" --require-approval never

if [ $? -eq 0 ]; then
    echo "   ✅ Deployment completed successfully!"
    
    # Clean up backup files
    rm -f lambda/enhanced_unified_model_backup.pkl
    rm -f lambda/enhanced_runtime_model_backup.pkl
    
    echo ""
    echo "🎉 Retraining pipeline completed successfully!"
    echo "   - New models trained and validated"
    echo "   - Lambda function updated"
    echo "   - API ready with improved predictions"
else
    echo "   ❌ Deployment failed!"
    exit 1
fi