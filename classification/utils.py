import os.path
import pickle
import string
import time

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

stemmer = SnowballStemmer("english")

COMPATIBILITY = 'COMPATIBILITY'
DEVICE = 'DEVICE'
ANDROID_VERSION = 'ANDROID VERSION'
HARDWARE = 'HARDWARE'

USAGE = 'USAGE'
APP_USABILITY = 'APP USABILITY'
UI = 'UI'

RESOURCES = 'RESOURCES'
PERFORMANCE = 'PERFORMANCE'
BATTERY = 'BATTERY'
MEMORY = 'MEMORY'

PRICING = 'PRICING'
LICENSING = 'LICENSING'
PRICE = 'PRICE'

PROTECTION = 'PROTECTION'
SECURITY = 'SECURITY'
PRIVACY = 'PRIVACY'

ERROR = 'ERROR'

OTHER = 'OTHER'

cat_subcat_dict = {
    COMPATIBILITY: [DEVICE, ANDROID_VERSION, HARDWARE],
    USAGE: [APP_USABILITY, UI],
    RESOURCES: [PERFORMANCE, BATTERY, MEMORY],
    PRICING: [LICENSING, PRICE],
    PROTECTION: [SECURITY, PRIVACY],
    ERROR: [ERROR],
}


def preprocess_review(review):
    # remove punctuation
    exclude = set(string.punctuation)
    review = ''.join(ch for ch in review if ch not in exclude)
    # remove stop words
    filtered_words = [word for word in review.split(' ') if word not in stopwords.words('english')]
    # apply stemming
    return ' '.join([stemmer.stem(word) for word in filtered_words])


def create_preprocessing_pipeline(text_data):
    text_prep = Pipeline([("vect", CountVectorizer(min_df=5, ngram_range=(1, 3), stop_words="english")),
                          ("tfidf", TfidfTransformer(norm=None))])
    text_prep.fit(text_data)
    return text_prep


def load_and_save_review_data(filepath, review_field, saved_filepath):
    data = pd.read_csv(filepath, encoding="ISO-8859-1")
    data["prep_" + review_field] = data[review_field].apply(lambda review: preprocess_review(review))
    data = data.sample(frac=1).reset_index(drop=True)
    joblib.dump(data, saved_filepath)
    return data


def get_cached_filepath(filepath):
    filename = os.path.split(filepath)[-1][:-4] + "_prep.pkl"
    return os.path.join(".", "preprocessed_files", filename)


def preprocess_review_data(filepath, review_field, text_prep=None):
    cached_filepath = get_cached_filepath(filepath)
    if os.path.isfile(cached_filepath):
        data = joblib.load(cached_filepath)
        return data, text_prep.transform(data["prep_" + review_field]), text_prep

    data = load_and_save_review_data(filepath, review_field, cached_filepath)

    if text_prep:
        return data, text_prep.transform(data["prep_" + review_field]), text_prep

    text_prep = create_preprocessing_pipeline(data[review_field])
    return data, text_prep.transform(data[review_field]), text_prep


def load_or_evaluate_classification(filepath, review_field, categories, cached, k):
    if not cached or not os.path.isfile(os.path.join(".", "internal_data", "results.pkl")):
        results = evaluate_classification(filepath, review_field, categories, k)
    else:
        pkl_file = open(os.path.join(".", "internal_data", "results.pkl"), 'rb')
        results = pickle.load(pkl_file)
    return results


def evaluate_classification(filepath, review_field, categories, k):
    all_results = {}
    for category in categories:
        category = "IS_" + category
        print("-----------------------------------")
        print("For category: %s" % category)
        data, X, _ = preprocess_review_data(filepath=filepath, review_field=review_field)
        y = data[category]
        splitter = StratifiedShuffleSplit(n_splits=k, test_size=0.2, random_state=0)

        results = {
            "f1_score": [],
            "precision": [],
            "recall": []
        }

        clf = ensemble.GradientBoostingClassifier(verbose=2, n_estimators=50)

        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y.iloc[train_idx], y.iloc[test_idx]
            clf.fit(X_train.toarray(), y_train)
            y_pred = clf.predict(X_test.toarray())
            results["f1_score"].append(f1_score(y_test, y_pred, average="macro"))
            results["precision"].append(precision_score(y_test, y_pred, average="macro"))
            results["recall"].append(precision_score(y_test, y_pred, average="macro"))
            print(classification_report(y_test, y_pred))
            print("precision: %.2f recall: %.2f f1_score: %.2f" % (precision_score(y_test, y_pred, average="macro"),
                                                                   recall_score(y_test, y_pred, average="macro"),
                                                                   f1_score(y_test, y_pred, average="macro")))

        for score, values in results.iteritems():
            results[score] = sum(values) / len(values)
        all_results[category] = results
        pkl_file = open(os.path.join(".", "internal_data", "results.pkl"), 'wb')
        pickle.dump(results, pkl_file, -1)
    return all_results


def train_classifier(clf, X_train, y_train):
    clf.fit(X_train.toarray(), y_train)
    return clf


def train_and_save_models(filepath, review_field, categories):
    data, X, text_prep = preprocess_review_data(filepath=filepath, review_field=review_field)
    joblib.dump(text_prep, os.path.join(".", "internal_data", "text_prep.pkl"))
    for category in categories:
        category = "IS_" + category
        print("Training %s classifier..." % category)
        clf = train_classifier(ensemble.GradientBoostingClassifier(verbose=2, n_estimators=500), X, data[category])
        model_details = {"text_field": review_field, "category": category}
        directory = os.path.join(".", "internal_data", category)
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(clf, os.path.join(directory, "model.pkl"))
        joblib.dump(model_details, os.path.join(directory, "model_details.pkl"))


def build_categories_list():
    categories_lists = cat_subcat_dict.values()
    categories = []
    for category_list in categories_lists:
        categories.extend(category_list)
    return ["IS_" + category for category in categories]


def build_pretty_categories_list():
    return [category[3:].lower().title() for category in build_categories_list()]


def preprocess_category_name(category):
    lower_category = category[3:].lower()
    return "_".join(lower_category.split())


def classify_and_save_results(filepath, text_field, categories):
    time_1 = time.time()
    text_prep = joblib.load(os.path.join(".", "classification", "internal_data", "text_prep.pkl"))
    diff = time.time() - time_1
    time_1 = time.time()
    print("Loading text_prep: %.2f seconds" % diff)
    data, X, _ = preprocess_review_data(filepath, text_field, text_prep)
    diff = time.time() - time_1
    time_1 = time.time()
    print("Preprocessing: %.2f seconds" % diff)
    pred_categories = []
    for category in categories:
        print("Category: %s" % category)

        directory = os.path.join(".", "classification", "internal_data", category)
        clf = joblib.load(os.path.join(directory, "model.pkl"))
        diff = time.time() - time_1
        time_1 = time.time()
        print("Loading text_prep: %.2f seconds" % diff)
        y_pred = clf.predict(X.toarray())
        data["PREDICTED_" + category] = [preprocess_category_name(category) if pred == 1 else "" for pred in y_pred]
        pred_categories.append("PREDICTED_" + category)
        diff = time.time() - time_1
        time_1 = time.time()
        print("Predicting %.2f" % diff)
    data["predictions"] = data[pred_categories].apply(lambda cats: [cat for cat in cats if cat], axis=1)
    new_filepath = filepath[:-4] + "_4.csv"
    data.to_csv(new_filepath, encoding="ISO-8859-1", index=False)
    diff = time.time() - time_1
    print("Saving csv file: %.2f" % diff)
    return data