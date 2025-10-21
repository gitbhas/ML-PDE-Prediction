from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
    aws_apigateway as apigateway,
    CfnOutput,
    Duration
)
from constructs import Construct

class PredictionLambdaStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create Lambda function using container image
        prediction_lambda = _lambda.Function(
            self, 'PredictionFunction',
            runtime=_lambda.Runtime.FROM_IMAGE,
            handler=_lambda.Handler.FROM_IMAGE,
            code=_lambda.Code.from_asset_image('./lambda'),
            timeout=Duration.seconds(300),
            memory_size=2048,
            environment={
                'PYTHONPATH': '/var/task'
            }
        )

        # Create API Gateway
        api = apigateway.RestApi(
            self, 'PredictionApi',
            rest_api_name='ML Prediction Service',
            description='API Gateway for ML Predictions',
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,
                allow_methods=apigateway.Cors.ALL_METHODS,
                allow_headers=['Content-Type', 'Authorization']
            )
        )

        # Create Lambda integration
        lambda_integration = apigateway.LambdaIntegration(
            prediction_lambda,
            request_templates={
                "application/json": '{ "statusCode": "200" }'
            }
        )

        # Add /predict resource
        predict_resource = api.root.add_resource('predict')
        predict_resource.add_method('GET', lambda_integration)
        predict_resource.add_method('POST', lambda_integration)

        # Output the API Gateway URL
        CfnOutput(
            self, 'PredictionApiUrl',
            value=f"{api.url}predict",
            description='Prediction API endpoint URL'
        )

        CfnOutput(
            self, 'LambdaFunctionName',
            value=prediction_lambda.function_name,
            description='Lambda function name'
        )