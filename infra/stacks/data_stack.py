from aws_cdk import (
    Stack, RemovalPolicy,
    aws_timestream as ts,
    aws_iam as iam,
)
from constructs import Construct
from dataclasses import dataclass

@dataclass
class DataProps:
    db_name:   str
    tbl_name:  str
    model_id:  str

class DataStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        db  = ts.CfnDatabase(self, "VitalsDB",
                             database_name="vitals_db")
        tbl = ts.CfnTable(self, "VitalsTable",
                          database_name=db.database_name,
                          table_name="vitals_ts",
                          retention_properties={"MemoryStoreRetentionPeriodInHours": 720,
                                                "MagneticStoreRetentionPeriodInDays": 365})
        tbl.node.add_dependency(db)  # order matters

        # IAM policy that lets Lambda query Timestream & invoke Bedrock
        self.lambda_policy = iam.ManagedPolicy(self, "LambdaPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["timestream:DescribeEndpoints",
                             "timestream:SelectValues"],
                    resources=["*"]),
                iam.PolicyStatement(
                    actions=["bedrock:InvokeModel"],
                    resources=["*"])
            ])

        self.out_props = DataProps(db_name=db.database_name,
                                   tbl_name=tbl.table_name,
                                   model_id="anthropic.claude-3-haiku-20240307-v1:0")
