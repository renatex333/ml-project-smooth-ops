import os
import pandas as pd
import logging
import joblib  
from process import load_data  
from sklearn.metrics import mean_squared_error

DATA_FOLDER = os.path.relpath("data", os.getcwd())
MODEL_FOLDER = os.path.relpath("models", os.getcwd())
LOGS_FOLDER = os.path.relpath("logs", os.getcwd())

def predict():
    model_file_path = os.path.join(MODEL_FOLDER, "model.pkl")
    model = joblib.load(model_file_path)

    scaler_file_path = os.path.join(MODEL_FOLDER, "scaler.pkl")
    scaler = joblib.load(scaler_file_path)

    data_path = os.path.join(DATA_FOLDER, "winequality-predict.csv")
    data = load_data(data_path)
    
    X = data.drop(columns=["quality"])
    X_scaled = scaler.fit_transform(X)

    predictions = model.predict(X_scaled)

    comparison = pd.DataFrame({
        "Actual Quality": data["quality"],
        "Predicted Quality": predictions
    })

    file_path = os.path.join(DATA_FOLDER, "predictions.csv")
    comparison.to_csv(file_path, index=False)
    logging.info(f"Comparison saved to '{file_path}'.")

if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-18s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%y-%m-%d %H:%M",
        filename=os.path.join(LOGS_FOLDER, "predict.log"),
        filemode="a",
    )

    predict()
