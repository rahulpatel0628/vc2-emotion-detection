import numpy as np
import pandas as pd
import pickle
import os
import logging
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_model(path: str):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_data(path: str):
    try:
        df = pd.read_csv(path)
        logging.info("Test data loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_data(df: pd.DataFrame):
    try:
        X = df.iloc[:, 0:-1].values
        y = df.iloc[:, -1].values
        logging.info("Test features prepared")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def evaluate_model(model, X, y):
    try:
        y_pred = model.predict(X)

        metrics = {
            "f1_score": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred)
        }

        logging.info(f"Evaluation Metrics: {metrics}")
        return metrics

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics: dict, path="reports/metrics.json"):
    try:
        os.makedirs(path, exist_ok=True)

        # Save JSON (for DVC)
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        logging.info("Metrics saved successfully")

    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def main():
    try:
        model = load_model("models/model.pkl")  # updated path
        test_df = load_data("data/features/test_bow.csv")

        X_test, y_test = prepare_data(test_df)

        metrics = evaluate_model(model, X_test, y_test)

        save_metrics(metrics)

        logging.info("Model evaluation completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()