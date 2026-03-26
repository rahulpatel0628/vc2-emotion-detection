import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

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

def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logging.info("Data loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["tweet_id"], inplace=True)

        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

        final_df['sentiment'] = final_df['sentiment'].replace({
            'happiness': 1,
            'sadness': 0
        })

        logging.info("Data preprocessing completed")
        return final_df
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise


def split_data(df: pd.DataFrame, test_size: float):
    try:
        train_data, test_data = train_test_split(
            df, test_size=test_size, random_state=42
        )
        logging.info("Train-test split completed")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error in splitting data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, path: str):
    try:
        os.makedirs(path, exist_ok=True)

        train_path = os.path.join(path, "train.csv")
        test_path = os.path.join(path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logging.info(f"Data saved at {path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


def main():
    try:
        params = load_params("params.yaml")
        test_size = params['data_ingestion']['test_size']

        url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"

        df = load_data(url)
        processed_df = preprocess_data(df)
        train_data, test_data = split_data(processed_df, test_size)

        save_data(train_data, test_data, os.path.join("data", "raw"))

        logging.info("Data ingestion pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()