# Source Folder

Inside this folder you will find the code pertaining to the implementation of the model and the visualization of data being studied upon.

## Files

1. `create_api.py`: Script used to create an API dashboard using AWS.
2. `create_function.py`: Script used to create function inside AWS Elastic Container Registry (ECR) repository.
3. `create_repository.py`: Script used for creating repository in AWS Elastic Container Registry (ECR).
4. `lambda_function.py`: Production version script for Prediction split in the Machine Learning process.
5. `predict.py`: Prediction split for the Machine Learning process. Run this after `train.py`.
6. `process.py`: File created to read the database being used, in this case, a .csv file.
7. `separate_data.py`: Script to separate original database into a training/predict split.
8. `train.py`: Training split for the Machine Learning process. Run this before `predict.py`.

## How to use the scripts

### Installing Dependencies

To install the project dependencies, use the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Train and Evaluate Models

### Separate data between training and prediction splits

To separate data between training and prediction, run:

```bash
python3 src/separate_data.py
```

It is important to separate data so the user can learn patterns on one dataset while its performance is independently validated on unseen data. This ensures accurate evaluation and validation of model quality.

### Train data

To train the dataset, run:

```bash
python3 src/train.py
```

This script trains the data in the training split and, by using three different models, chooses the best one in regards to performance for further use.

### Predict data

To make predictions on top of the dataset, run:

```bash
python3 src/predict.py
```

After predictions are done, a Ground Truth Evaluation (GTE) is made. It involves comparing the predicted outputs of a model to the actual, known values (ground truth) in order to assess the model's accuracy and performance.

### Check out the MLflow dashboard

To check out the metrics and artifacts generated during training, run MLflow with:

```bash
mlflow ui -p 5005
```

And access the dashboard at [localhost:5005](localhost:5005).

## Deploy model in AWS

### Configure your AWS CLI

Having [AWS CLI installed](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), configure your credentials on a profile:

```bash
aws configure --profile mlops
```

To set it as deafult profile:

Linux:

```bash
export AWS_PROFILE=mlops
```

Windows CMD:

```bash
set AWS_PROFILE=mlops
```

Windows PowerShell:

```bash
env:AWS_PROFILE = "mlops"
```

### Create and Configure ECR repository

To set up the project, let's build the Docker image from the `Dockerfile` with the `test` tag.

```bash
docker build --platform linux/amd64 -t smooth-ops-project-image:test .
```

Then run the following script to create a repository in AWS ECR:

> [!NOTE]  
> Save the `REPOSITORY_URI`.

```bash
python3 src/create_repository.py
```

Then login to ECR using the Docker CLI:

> [!IMPORTANT]  
> Substitute `AWS_ACCOUNT_ID` for your AWS Account ID.

```bash
aws ecr get-login-password --profile mlops --region us-east-2 | docker login --username AWS --password-stdin AWS_ACCOUNT_ID.dkr.ecr.us-east-2.amazonaws.com
```

Rebuild your Docker image (if needed), tag your local Docker image (`Dockerfile`) into the repository as the latest version and push the image:

> [!IMPORTANT]  
> Substitute `REPOSITORY_URI` for the correct repository's URI.

```bash
docker build --platform linux/amd64 -t smooth-ops-project-image:test .

docker tag smooth-ops-project-image:test REPOSITORY_URI:latest

docker push REPOSITORY_URI:latest
```

### Create Lambda Function

To create a Lambda function from the ECR image, run:

```bash
python3 src/create_function.py
```

### Create API Gateway

To create an API Gateway that exposes the Lambda function, run:

```bash
python3 src/create_api.py
```

#### Local Testing

To test the function locally, use:

```bash
pytest
```

#### Remote Testing

After deploying the Lambda function and API Gateway, you can verify the setup by running:

```bash
pytest --local
```

The `--local` flag ensures that tests requiring local resources, such as environment variables, are executed.

## References

- [AWS Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [AWS Lambda Layers Documentation](https://docs.aws.amazon.com/lambda/latest/dg/chapter-layers.html)
- [API Gateway Documentation](https://docs.aws.amazon.com/apigateway/)
- [Pytest Documentation](https://docs.pytest.org/en/stable/)
