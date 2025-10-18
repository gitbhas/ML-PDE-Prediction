import aws_cdk as core
import aws_cdk.assertions as assertions

from cdkapp.cdkapp_stack import CdkappStack

# example tests. To run these tests, uncomment this file along with the example
# resource in cdkapp/cdkapp_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = CdkappStack(app, "cdkapp")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
