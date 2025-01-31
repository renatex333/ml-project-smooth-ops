"""
Module to create a new AWS Lambda function from ECR image
"""

import os
import boto3
import random
import string
import logging
from dotenv import load_dotenv, set_key, unset_key

LOGS_FOLDER = os.path.relpath("logs", os.getcwd())

def main():
    load_dotenv()

    function_name = os.getenv("FUNCTION_NAME")

    # Provide Image URI
    version = "latest"
    image_uri = f"{os.getenv('REPOSITORY_URI')}:{version}"

    lambda_role_arn = os.getenv("AWS_LAMBDA_ROLE_ARN")

    lambda_client = boto3.client(
        "lambda",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    try:
        lambda_client.delete_function(FunctionName=function_name)
        logging.info("Existing Function was Deleted!")
        unset_key(".env", "FUNCTION_ARN")
    except lambda_client.exceptions.ResourceNotFoundException:
        logging.warning("Function does not exist.")

    logging.info("Creating Lambda Function: %s", function_name)
    response = lambda_client.create_function(
        FunctionName=function_name,
        PackageType="Image",
        Code={"ImageUri": image_uri},
        Role=lambda_role_arn,
        Timeout=30,  # Optional: function timeout in seconds
        MemorySize=128,  # Optional: function memory size in megabytes
    )

    id_num = "".join(random.choices(string.digits, k=7))

    api_gateway_permissions = lambda_client.add_permission(
        FunctionName=function_name,
        StatementId="api-gateway-permission-statement-" + id_num,
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
    )

    function_arn = response["FunctionArn"]
    logging.info("Lambda Function %s Created Successfully!", function_name)
    print(f"Function ARN: {function_arn}")
    set_key(".env", "\nFUNCTION_ARN", function_arn)

if __name__ == "__main__":
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-18s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%y-%m-%d %H:%M",
        filename=os.path.join(LOGS_FOLDER, f"{script_name}.log"),
        filemode="w",
    )
    main()
