#!/usr/bin/env python3
from aws_cdk import App
from stacks.prediction_lambda_stack import PredictionLambdaStack

app = App()
PredictionLambdaStack(app, "PredictionLambdaStack")
app.synth()