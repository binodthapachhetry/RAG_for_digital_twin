# local_runner.py
import json
import os
from handler import handler

# OPTIONAL: avoid real Bedrock calls while iterating locally
os.environ.setdefault("LOCAL_TEST", "1")  # add a small if-check in handler (see note)

with open("/Users/binod/Downloads/payload.json") as f:
    payload = json.load(f)

event = {
    "rawPath": "/query",
    "headers": {"content-type": "application/json"},
    "requestContext": {"http": {"method": "POST"}},
    "body": json.dumps(payload),
    "isBase64Encoded": False,
}

resp = handler(event, None)
print(resp["statusCode"])
print(resp["body"])
