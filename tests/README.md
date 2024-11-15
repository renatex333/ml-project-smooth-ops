# Tests Folder

Inside this folder you will find the tests used to check if the functionalities are working according to plan.

## Files

1. conftest.py : Handles testing and loading of .env variables for your environment configuration.
2. test_aws.py : Tests that ensure the Lambda function and API Gateway deployment on AWS are done correctly.
3. test_function.py : Tests to verify the correct response from the developed Lambda function and that the models are loaded and behave as expected.

## How to Run the code correctly

In the Command Line Interface (CLI), run:

    pytest

Or run the following command to run both code test and aws tests by using local variables:

    pytest --local

After running the test_function.py test, check if lambda function returned is accurate and if models are loaded correctly.

After running the test_aws.py test, check if deploy for the lambda function and if the API gateway are correctly displayed in AWS.
