#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.api_stack import ApiStack

app = cdk.App()

# With Timestream temporarily disabled we do not deploy DataStack.
api = ApiStack(app, "ApiStack")

app.synth()
