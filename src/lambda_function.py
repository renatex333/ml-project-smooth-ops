import os
import json
import pickle
import pandas as pd

def loader(object_name: str):
    """
    Function to load a model or encoder from the models folder
    """
    if not isinstance(object_name, str):
        raise ValueError("Object path is not a string")
    models_folder = os.path.relpath("models", os.getcwd())
    object_path = f"{models_folder}/{object_name}.pkl"
    try:
        with open(object_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Invalid object name. The object at {object_path} does not exist.")

def predict(event, context):
    """
    Handler function to make predictions
    """
    if "body" not in event:
        return {
            "message": "No body in the request",
            "error": "None",
            "prediction": "None"
        }

    try:
        body = json.loads(event["body"])
        
        model = loader("model")
        encoder = loader("encoder")

        df_person = pd.DataFrame([body])
        person_t = encoder.transform(df_person)
        pred = model.predict(person_t)[0]
    except json.JSONDecodeError as e:
        return {
            "message": "Invalid body in the request",
            "error": f"{type(e).__name__}: {str(e)}",
            "prediction": "None"
        }
    except (ValueError, FileNotFoundError) as e:
        return {
            "message": f"Error loading the model",
            "error": f"{type(e).__name__}: {str(e)}",
            "prediction": "None"
        }
    except Exception as e:
        return {
            "message": f"Invalid body in the request",
            "error": f"{type(e).__name__}: {str(e)}",
            "prediction": "None"
        }
    else:
        return {
            "message": "Prediction made successfully",
            "error": "None",
            "prediction": str(pred)
        }
