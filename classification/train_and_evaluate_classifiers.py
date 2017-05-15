from utils import *
import pandas as pd

REVIEW_FIELD = "reviewText"

DATA_REVIEWS_CSV = "./data/reviews.csv"

TRAINING_DATA_PATH = "./data/reviews.csv"


def merge_training_data():
    if not os.path.exists(TRAINING_DATA_PATH):
        data_1 = pd.read_csv("./data/golden_set_clean.csv")
        data_2 = pd.read_csv("./data/train_reviews_manual.csv")
        data = pd.concat([data_1, data_2])
        data.drop_duplicates("reviewText", inplace=True)
        data.to_csv(TRAINING_DATA_PATH)


def build_subcategories_list():
    all_subcategories = []
    for subcategories in cat_subcat_dict.values():
        all_subcategories.extend(subcategories)
    return all_subcategories


def run_evaluation(force_train=False):
    if force_train or not os.path.isfile("./internal_data/cross_validation_results.pkl"):
        subcategories = build_subcategories_list()
        results = load_or_evaluate_classification(DATA_REVIEWS_CSV, REVIEW_FIELD, subcategories, False, 5)
        joblib.dump(results, "./internal_data/cross_validation_results.pkl")
    else:
        results = joblib.load("./internal_data/cross_validation_results.pkl")
    for key, values in results.iteritems():
        print("%s: %s" % (key, values))


def train_models():
    subcategories = build_subcategories_list()
    train_and_save_models(DATA_REVIEWS_CSV, REVIEW_FIELD, subcategories)


def main():
    merge_training_data()
    run_evaluation()
    train_models()


if __name__ == "__main__":
    main()
