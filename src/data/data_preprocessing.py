import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logging.info("NLTK resources downloaded")
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {e}")
        raise

def load_data(train_path: str, test_path: str):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Data loaded successfully")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return "".join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def remove_punctuation(text):
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()

def remove_urls(text):
    return re.sub(r"http\S+", "", text).strip()


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Starting text normalization")

        df['content'] = df['content'].astype(str)

        df['content'] = df['content'].apply(remove_urls)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_punctuation)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(lemmatization)

        logging.info("Text normalization completed")
        return df

    except Exception as e:
        logging.error(f"Error in text normalization: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, path: str):
    try:
        os.makedirs(path, exist_ok=True)

        train_path = os.path.join(path, "train_processed.csv")
        test_path = os.path.join(path, "test_processed.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Processed data saved at {path}")

    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    try:
        download_nltk_resources()

        train_df, test_df = load_data(
            "./data/raw/train.csv",
            "./data/raw/test.csv"
        )

        train_processed = normalize_text(train_df)
        test_processed = normalize_text(test_df)

        save_data(train_processed, test_processed, "data/processed")

        logging.info("Feature engineering pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()