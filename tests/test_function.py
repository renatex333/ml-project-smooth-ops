import json
import pytest
import lambda_function

def load_valid_payload():
    """
    Return a payload to test the function and the API.
    """
    payload = json.dumps({
        "fixed acidity": 7.7,
        "volatile acidity": 0.56,
        "citric acid": 0.08,
        "residual sugar": 2.5,
        "chlorides": 0.114,
        "free sulfur dioxide": 14.0,
        "total sulfur dioxide": 46.0,
        "density": 0.9971,
        "pH": 3.24,
        "sulphates": 0.66,
        "alcohol": 9.6,
    })
    return payload

def load_invalid_payload():
    """
    Return an invalid payload to test the function and the API.
    """
    payload = json.dumps({
        "fixed_acidity": 7.7,
    })
    return payload

def test_no_body_in_request():
    """
    Tests the lambda function with no body in the request
    """
    event = {}

    expected = {
        "message": "No body in the request",
        "error": "None",
        "prediction": "None"
    }

    # Test if return is as expected
    assert lambda_function.predict(event, None) == expected

def test_invalid_body():
    """
    Tests the lambda function with invalid body in the request
    """
    event = {
        "body": "invalid"
    }

    response = lambda_function.predict(event, None)
    assert response["error"] != "None", f"Error is None. Should be an error message. Message: {response['message']}"

def test_invalid_payload():
    """
    Tests the lambda function with invalid payload in the request
    """
    payload = load_invalid_payload()
    event = {
        "body": payload
    }

    response = lambda_function.predict(event, None)
    assert response["error"] != "None", f"Error is None. Should be an error message. Message: {response['message']}"

def test_valid_payload():
    """
    Tests the lambda function with valid payload in the request
    """
    payload = load_valid_payload()
    event = {
        "body": payload
    }

    response = lambda_function.predict(event, None)
    assert response["prediction"] != "None", f"Prediction is None. Should be a prediction. Error: {response['error']}"

def test_loader_invalid_input():
    """
    Tests the loader function with invalid input
    """
    with pytest.raises(FileNotFoundError):
        lambda_function.loader("invalid_model")

def test_loader_valid_input():
    """
    Tests the loader function with valid input
    """
    model = lambda_function.loader("model")
    scaler = lambda_function.loader("scaler")

    assert model is not None, "Model is None. Should be a model."
    assert scaler is not None, "Scaler is None. Should be an scaler."
