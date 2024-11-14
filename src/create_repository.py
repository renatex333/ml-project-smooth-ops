"""
Module to create a repository in ECR
"""

import os
import boto3
import logging
from dotenv import load_dotenv, set_key, unset_key

LOGS_FOLDER = os.path.relpath("logs", os.getcwd())

def main():
    load_dotenv()

    repository_name = os.getenv("REPOSITORY_NAME")

    ecr_client = boto3.client(
        "ecr",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    try:
        ecr_client.delete_repository(repositoryName=repository_name, force=True)
        logging.info("Existing Repository was Deleted!")
        unset_key(".env", "REPOSITORY_URI")
        unset_key(".env", "REPOSITORY_ARN")
    except ecr_client.exceptions.RepositoryNotFoundException:
        logging.warning("Repository not found")

    logging.info(f"Creating Repository: {repository_name}")
    response = ecr_client.create_repository(
        repositoryName=repository_name,
        imageScanningConfiguration={"scanOnPush": True},
        imageTagMutability="MUTABLE",
    )

    repository_uri = response["repository"]["repositoryUri"]
    repository_arn = response["repository"]["repositoryArn"]
    logging.info(f"Repository {repository_name} Created Successfully!")
    print(f"Repository URI: {repository_uri}")
    print(f"Repository ARN: {repository_arn}")
    set_key(".env", "\nREPOSITORY_URI", repository_uri)
    set_key(".env", "\nREPOSITORY_ARN", repository_arn)

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
