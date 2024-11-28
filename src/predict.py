import os
import joblib
import logging
import pandas as pd
from process import load_data
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_FOLDER = os.path.relpath("data", os.getcwd())
MODEL_FOLDER = os.path.relpath("models", os.getcwd())
IMAGES_FOLDER = os.path.relpath("img", os.getcwd())
LOGS_FOLDER = os.path.relpath("logs", os.getcwd())

def predict():
    model_file_path = os.path.join(MODEL_FOLDER, "model.pkl")
    model = joblib.load(model_file_path)

    scaler_file_path = os.path.join(MODEL_FOLDER, "scaler.pkl")
    scaler = joblib.load(scaler_file_path)

    data_path = os.path.join(DATA_FOLDER, "winequality-predict.csv")
    data = load_data(data_path)
    
    X = data.drop(columns=["quality"])
    ground_truth = data["quality"]
    X_scaled = scaler.fit_transform(X)

    predictions = model.predict(X_scaled)

    comparison = pd.DataFrame({
        "Actual Quality": ground_truth,
        "Predicted Quality": predictions
    })

    file_path = os.path.join(DATA_FOLDER, "predictions.csv")
    comparison.to_csv(file_path, index=False)
    logging.info("Comparison saved to '%s'.", file_path)
    evaluate(comparison, model.classes_)

def evaluate(comparison: pd.DataFrame, model_classes: list):
    # Calculate metrics
    accuracy = accuracy_score(comparison["Actual Quality"], comparison["Predicted Quality"])
    precision = precision_score(comparison["Actual Quality"], comparison["Predicted Quality"], average="weighted", zero_division=0)
    recall = recall_score(comparison["Actual Quality"], comparison["Predicted Quality"], average="weighted", zero_division=0)
    f1 = f1_score(comparison["Actual Quality"], comparison["Predicted Quality"], average="weighted", zero_division=0)

    logging.info("Ground Truth Evaluation - Accuracy: %f:.2f", accuracy)
    logging.info("Ground Truth Evaluation - Precision: %f:.2f", precision)
    logging.info("Ground Truth Evaluation - Recall: %f:.2f", recall)
    logging.info("Ground Truth Evaluation - F1 Score: %f:.2f", f1)
    conf_mat = confusion_matrix(comparison["Actual Quality"], comparison["Predicted Quality"], labels=model_classes)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model_classes
    )
    fig, ax = plt.subplots()
    conf_mat_disp.plot(ax=ax)
    fig_name = "confusion_matrix_ground_truth.png"
    fig_path = os.path.join(IMAGES_FOLDER, fig_name)
    fig.savefig(fig_path)
    logging.info("Confusion matrix saved to '%s'.", fig_path)

if __name__ == "__main__":
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-18s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%y-%m-%d %H:%M",
        filename=os.path.join(LOGS_FOLDER, f"{script_name}.log"),
        filemode="w",
    )
    predict()
