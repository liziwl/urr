from utils import *
from classification_utils import *
import pandas as pd
import time

REVIEW_FIELD = "reviewText"
TRAINING_PATH = "./data/all_reviews_data.csv"


def analyze_data():
    data = pd.read_csv(TRAINING_PATH, encoding="ISO-8859-1", error_bad_lines=False)
    print(len(data))
    for stopword in stopwords.words('english'):
        print(stopword)


def predict(clf, data):
    return clf.predict(data)


def get_ending(csv=True):
    return ".csv" if csv else ".txt"


def get_evaluation_filename(n_estimators, suffix="", csv=True):
    now = time.strftime("%c")
    now = "_".join(now.split())
    now = "_".join(now.split(":"))
    return "./evaluation_results/evaluation_results_" + str(n_estimators) + "_" + now + suffix + get_ending(csv)


def main():
    subcategories = build_categories_list()
    # subcategories = ["IS_MEMORY"]
    n_estimators = 200
    clf = ensemble.GradientBoostingClassifier(verbose=2, learning_rate=0.01, n_estimators=n_estimators)
    suffix = "all_data_all"
    results = evaluate_classification(TRAINING_PATH, get_evaluation_filename(n_estimators, suffix, csv=False),
                                      get_evaluation_filename(n_estimators, suffix), REVIEW_FIELD,
                                      subcategories, 10, clf, predict, "..")
    for key, values in results.iteritems():
        print("%s: %s" % (key, values))


if __name__ == "__main__":
    main()
