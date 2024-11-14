import os
import logging
from process import load_data
from sklearn.model_selection import train_test_split

DATA_FOLDER = os.path.relpath("data", os.getcwd())
LOGS_FOLDER = os.path.relpath("logs", os.getcwd())

def main():
    data_path = os.path.join(DATA_FOLDER, "winequality-red.csv")
    data = load_data(data_path)

    train_data, predict_data = train_test_split(data, test_size=0.25, random_state=42, shuffle=True)

    train_data.to_csv(os.path.join(DATA_FOLDER, "winequality-train.csv"), index=False)
    predict_data.to_csv(os.path.join(DATA_FOLDER, "winequality-predict.csv"), index=False)
    logging.info("Data separated into train and predict datasets.")

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