# Source Folder

Inside this folder you will find the code pertaining to the implementation of the model and the visualization of data being studied upon.

## Files

1. create_api.py : Script used to create an API dashboard using AWS.
2. create_function.py : Script used to create function inside AWS Elastic Container Registry (ECR) repository.
3. create_repository.py : Script used for creating repository in AWS Elastic Container Registry (ECR).
4. deploy.py : Script to deploy model using aws lambda.
5. lambda_function.py : Production version script for Prediction split in the Machine Learning process.
6. predict.py : Prediction split for the Machine Learning process. Run this after train.py.
7. process.py : File created to read the database being used, in this case, a .csv file.
8. separate_data.py: Script to separate original database into a training/predict split.
9. train.py : Training split for the Machine Learning process. Run this before predict.py.

## How to Run the code correctly
