import numpy as np
import pandas as pd
import pickle
import yaml
import logging
import os
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logging.info("Parameters loaded successfully")
        return params
    except Exception as e:
        logging.error(f"Error loading params: {e}")
        raise

def load_data(path: str):
    try:
        df = pd.read_csv(path)
        logging.info("Training data loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_data(df: pd.DataFrame):
    try:
        X = df.iloc[:, 0:-1].values
        y = df.iloc[:, -1].values
        logging.info("Features and labels prepared")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def train_model(X, y, n_estimators: int, learning_rate: float):
    try:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        model.fit(X, y)
        logging.info("Model training completed")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model, path="models/model.pkl"):
    try:
        os.makedirs("models", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved at {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main():
    try:
        params = load_params("params.yaml")['model_building']
        n_estimators = params['n_estimators']
        learning_rate = params['learning_rate']

        df = load_data("./data/features/train_bow.csv")

        X, y = prepare_data(df)

        model = train_model(X, y, n_estimators, learning_rate)

        save_model(model)

        logging.info("Model training pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()