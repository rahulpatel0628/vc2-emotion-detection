import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

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

def load_data(train_path: str, test_path: str):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Processed data loaded successfully")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        x_train = train_df['content'].values
        y_train = train_df['sentiment'].values

        x_test = test_df['content'].values
        y_test = test_df['sentiment'].values

        logging.info("Features and labels prepared")
        return x_train, y_train, x_test, y_test

    except Exception as e:
        logging.error(f"Error preparing features: {e}")
        raise

def apply_bow(x_train, x_test, max_features: int):
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        logging.info("Bag of Words transformation completed")
        return x_train_bow, x_test_bow

    except Exception as e:
        logging.error(f"Error in vectorization: {e}")
        raise

def create_feature_df(x_train_bow, y_train, x_test_bow, y_test):
    try:
        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        logging.info("Feature DataFrames created")
        return train_df, test_df

    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, path: str):
    try:
        os.makedirs(path, exist_ok=True)

        train_path = os.path.join(path, "train_bow.csv")
        test_path = os.path.join(path, "test_bow.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Feature data saved at {path}")

    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    try:
        params = load_params("params.yaml")
        max_features = params['feature_engineering']['max_features']

        train_df, test_df = load_data(
            "./data/processed/train_processed.csv",
            "./data/processed/test_processed.csv"
        )

        x_train, y_train, x_test, y_test = prepare_features(train_df, test_df)

        x_train_bow, x_test_bow = apply_bow(x_train, x_test, max_features)

        train_final, test_final = create_feature_df(
            x_train_bow, y_train, x_test_bow, y_test
        )

        save_data(train_final, test_final, "data/features")

        logging.info("Feature engineering pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()