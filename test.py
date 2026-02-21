from data import test
import boto3
import json

endpoint_name = "prod-endpoint"
region_name = "ap-northeast-2"
smrt = boto3.client("runtime.sagemaker", region_name=region_name)

test_data_json = json.dumps({"instances": test[:20].toarray().tolist()})

prediction = smrt.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=test_data_json,
    ContentType="application/json",
)

prediction = prediction["Body"].read().decode("ascii")
print(prediction)
