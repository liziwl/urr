from utils import *
from classification_utils import *
import pandas as pd
import time

REVIEW_FIELD = "reviewText"
REVIEWS_PATH = "./data/reviews_data.csv"
ALL_REVIEWS_PATH = "./data/all_reviews_data.csv"
MODELS_SAVE_PATH = "./best_models"

model_configurations = {
    "IS_DEVICE": {"data_path": REVIEWS_PATH, "with_stopwords": True, "n_estimators": 200},
    "IS_ANDROID VERSION": {"data_path": ALL_REVIEWS_PATH, "with_stopwords": False, "n_estimators": 200},
    "IS_HARDWARE": {"data_path": ALL_REVIEWS_PATH, "with_stopwords": False, "n_estimators": 200},
    "IS_APP USABILITY": {"data_path": REVIEWS_PATH, "with_stopwords": True, "n_estimators": 200},
    "IS_UI": {"data_path": REVIEWS_PATH, "with_stopwords": True, "n_estimators": 200},
    "IS_PERFORMANCE": {"data_path": ALL_REVIEWS_PATH, "with_stopwords": False, "n_estimators": 200},
    "IS_BATTERY": {"data_path": REVIEWS_PATH, "with_stopwords": True, "n_estimators": 200},
    "IS_MEMORY": {"data_path": ALL_REVIEWS_PATH, "with_stopwords": False, "n_estimators": 200},
    "IS_LICENSING": {"data_path": REVIEWS_PATH, "with_stopwords": True, "n_estimators": 200},
    "IS_PRICE": {"data_path": REVIEWS_PATH, "with_stopwords": True, "n_estimators": 200},
    "IS_SECURITY": {"data_path": REVIEWS_PATH, "with_stopwords": False, "n_estimators": 200},
    "IS_PRIVACY": {"data_path": REVIEWS_PATH, "with_stopwords": False, "n_estimators": 200},
    "IS_COMPLAINT": {"data_path": REVIEWS_PATH, "with_stopwords": False, "n_estimators": 200},
}


def train_classifiers():
    for category, model_config in model_configurations.iteritems():
        train_and_save_model(REVIEW_FIELD, category, model_config)


if __name__ == "__main__":
    train_classifiers()