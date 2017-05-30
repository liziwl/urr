from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import *


def x_predict(clf, data):
    y_pred = clf.predict(data)
    y_pred = [round(value) for value in y_pred]
    return y_pred


def predict(clf, data):
    return clf.predict(data.toarray())


def main():
    filepath = "./data/train_reviews_manual.csv"
    evaluate_classification(filepath, "./evaluation_results.txt", "reviewText", build_categories_list(), 5,
                            XGBClassifier(), x_predict, "..")
    evaluate_classification(filepath, "./evaluation_results.txt", "reviewText", build_categories_list(), 5,
                            ensemble.GradientBoostingClassifier(verbose=2, n_estimators=50), x_predict, "..")


def xtrain_and_save_models(filepath, review_field, categories):
    data, X, text_prep = preprocess_review_data(filepath=filepath, review_field=review_field)
    for category in categories:
        category = "IS_" + category
        print("Training %s classifier..." % category)


if __name__ == "__main__":
    main()