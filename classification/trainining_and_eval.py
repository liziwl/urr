from utils import *
from classification_utils import *
from train_and_evaluate_classifiers import *
import pandas as pd

REVIEW_FIELD = "reviewText"
TRAINING_PATH = "./data/train_reviews_manual_copy_04.csv"


def analyze_data():
    # how many apps
    # print: how many reviews per app
    # print: how many reviews per category
    pass


def predict(clf, data):
    return clf.predict(data)


def main():
    subcategories = build_categories_list()
    data = pd.read_csv(TRAINING_PATH)
    clf = ensemble.GradientBoostingClassifier(verbose=1, n_estimators=200)
    results = evaluate_classification(TRAINING_PATH, "./evaluation_results.txt", REVIEW_FIELD,
                                      subcategories, 10, clf, predict, "..")
    for key, values in results.iteritems():
        print("%s: %s" % (key, values))


if __name__ == "__main__":
    main()