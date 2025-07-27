# built-ins
import os
from pathlib import Path

# CDK
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as _lambda,
    aws_apigatewayv2_alpha as apigw,
    aws_apigatewayv2_integrations_alpha as integrations,
    aws_iam as iam, 
    CfnOutput,
)
from aws_cdk.aws_lambda_python_alpha import PythonFunction
from constructs import Construct

class ApiStack(Stack):
    """
    Public HTTP API backed by the TimeseriesAgent Lambda.
    Timestream is currently disabled, so this stack no longer depends
    on the DataStack (or its DataProps).
    """

    def __init__(self, scope: Construct, id: str, **kw) -> None:
        super().__init__(scope, id, **kw)

        # locate repo root: this file -> stacks -> infra -> (repo root)
        repo_root = Path(__file__).resolve().parents[2]
        src_path = repo_root / "src"

        if not src_path.exists():
            raise FileNotFoundError(f"Lambda source dir not found: {src_path}")

        # Obtain the Bedrock model identifier from CDK context or env var.
        model_id = (
            self.node.try_get_context("MODEL_ID")
            or os.environ.get("MODEL_ID", "")
        )

        # ── guard rail: stop synthesis/deploy if MODEL_ID is missing ──
        if not model_id:  # pragma: no cover
            raise ValueError(
                "Bedrock model identifier (MODEL_ID) not supplied.\n"
                "Provide it via CDK context, e.g.:\n"
                "    cdk deploy -c MODEL_ID=<bedrock-model-id>\n"
                "or export MODEL_ID=<bedrock-model-id> before running CDK."
            )

        fn = PythonFunction(
            self,
            "TimeseriesAgent",
            entry=str(src_path),
            index="handler.py",
            handler="handler",
            runtime=_lambda.Runtime.PYTHON_3_11,
            memory_size=512,
            timeout=Duration.seconds(10),
            environment={
                "MODEL_ID": model_id,
            },
        )

        # Only Bedrock permissions are needed while Timestream is disabled.
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],  # or specific model ARN
            )
        )

        api = apigw.HttpApi(
            self,
            "PublicAPI",
            cors_preflight=apigw.CorsPreflightOptions(
                allow_origins=["*"],
                allow_methods=[apigw.CorsHttpMethod.POST],
            ),
        )

        api.add_routes(
            path="/query",
            methods=[apigw.HttpMethod.POST],
            integration=integrations.HttpLambdaIntegration("Hook", fn),
        )

        self.api_url = api.api_endpoint
        CfnOutput(self, "ApiUrl", value=api.api_endpoint)

