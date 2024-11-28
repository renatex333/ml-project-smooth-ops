import os
import mlflow
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib  # Add this for saving/loading models
import logging
from dotenv import load_dotenv
from process import load_data

DATA_FOLDER = os.path.relpath("data", os.getcwd())
MODEL_FOLDER = os.path.relpath("models", os.getcwd())
IMAGES_FOLDER = os.path.relpath("img", os.getcwd())
LOGS_FOLDER = os.path.relpath("logs", os.getcwd())

def train_model():
    data_path = os.path.join(DATA_FOLDER, "winequality-train.csv")
    data = load_data(data_path)

    X = data.drop("quality", axis=1)
    y = data["quality"]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "SVC": SVC(),
        "SGDClassifier": SGDClassifier(),
        "RandomForestClassifier": RandomForestClassifier()
    }

    best_model = None
    best_score = 0

    for i, (model_name, model) in enumerate(models.items()):
        logging.info("Training %s...", model_name)
        model.fit(X_train, y_train)

        # Infer signature (input and output schema)
        signature = mlflow.models.signature.infer_signature(
            X_train, model.predict(X_train)
        )

        # Log model
        mlflow.sklearn.log_model(
            model,
            f"model_{model_name}",
            signature=signature,
            registered_model_name=f"wine-model-{model_name}",
            input_example=X_train[:3],
        )

        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        logging.info("%s Accuracy: %f:.2f", model_name, accuracy)
        if accuracy > best_score:
            best_model = model
            best_score = accuracy

        # Log metrics
        mlflow.log_metric("accuracy", accuracy, step=i)
        mlflow.log_metric("precision", precision, step=i)
        mlflow.log_metric("recall", recall, step=i)
        mlflow.log_metric("f1", f1, step=i)
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        fig, ax = plt.subplots()
        conf_mat_disp.plot(ax=ax)
        fig_name = f"confusion_matrix_{model_name}.png"
        fig_path = os.path.join(IMAGES_FOLDER, fig_name)
        fig.savefig(fig_path)
        mlflow.log_artifact(fig_path)
    
    # Save the best-performing model
    model_file_path = os.path.join(MODEL_FOLDER, "model.pkl")
    scaler_file_path = os.path.join(MODEL_FOLDER, "scaler.pkl")
    if best_model is not None:
        joblib.dump(best_model, model_file_path)
        joblib.dump(scaler, scaler_file_path)
        logging.info("Best model (%s) saved as '%s'.", type(best_model).__name__, model_file_path)
        logging.info("Scaler saved as '%s'.", scaler_file_path)
    
    # GridSearchCV for hyperparameter tuning
    # print(f"Hyperparameters of {type(best_model).__name__}: ", best_model.get_params())
    # param_grid = {
    #     "C": np.arange(0.1, 2, 0.1),
    #     "kernel":["linear", "rbf"],
    #     "gamma": np.arange(0.1, 2, 0.1)
    # }
    # grid = GridSearchCV(best_model, param_grid, cv=5)
    # grid.fit(X_train, y_train)
    # print(f"Best parameters for {type(best_model).__name__}:", grid.best_params_)
    # print(f"Best cross-validation score for {type(best_model).__name__}:", grid.best_score_)

    # # Save the GridSearchCV best estimator if desired
    # estimator_filename = "best_model_gridsearch.pkl"
    # estimator_file_path = os.path.join(MODEL_FOLDER, estimator_filename)
    # joblib.dump(grid.best_estimator_, estimator_file_path)
    # print(f"Best GridSearchCV model saved as '{estimator_file_path}'.")

def main(run_name: str = "wine-quality-model", experiment_name: str = "wine-quality-exp"):
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", run_name)
        train_model()


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

