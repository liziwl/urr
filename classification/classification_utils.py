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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

stemmer = SnowballStemmer("english")

REVIEW_FIELD = "reviewText"
REVIEWS_PATH = "./data/reviews_data.csv"
ALL_REVIEWS_PATH = "./data/all_reviews_data.csv"

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

COMPLAINT = 'COMPLAINT'

REVIEW_FIELD = "reviewText"
RATING_FIELD = "ratingStars"
USER_REVIEWS_HOME = "user_reviews_files"

cat_subcat_dict = {
    COMPATIBILITY: [DEVICE, ANDROID_VERSION, HARDWARE],
    USAGE: [APP_USABILITY, UI],
    RESOURCES: [PERFORMANCE, BATTERY, MEMORY],
    PRICING: [LICENSING, PRICE],
    PROTECTION: [SECURITY, PRIVACY],
    COMPLAINT: [COMPLAINT]
}

categories_definitions = {
    DEVICE: "mentions a specific mobile phone device (i.e Galaxy 6)",
    ANDROID_VERSION: "references the OS version (i.e. Marshmallow)",
    HARDWARE: "talks about a specific hardware component",
    APP_USABILITY: "talks about ease or difficulty in using a feature",
    UI: "mentions an UI element (i.e. button, menu item)",
    PERFORMANCE: "talks about the performance of the app (i.e. slow, fast)",
    BATTERY: "references related to the battery (i.e. drains the battery)",
    MEMORY: "mentions issues related to the memory (i.e. out of memory)",
    LICENSING: "references the licensing model of the app (i.e. free, pro version)",
    PRICE: "talks about money aspects (i.e. donated $5)",
    SECURITY: "talks about the security/lack of it",
    PRIVACY: "issues related to permissions and user data",
    COMPLAINT: "the users reports or complains about an issue with the app"
}


def preprocess_review(review, remove_stopwords=False):
    try:
        # remove punctuation
        exclude = set(string.punctuation)
        review = ''.join(ch for ch in review if ch not in exclude)
        # remove stop words
        if remove_stopwords:
            filtered_words = [word for word in review.split() if word not in stopwords.words('english')]
        else:
            filtered_words = review.split()
        # apply stemming
        return ' '.join([stemmer.stem(word) for word in filtered_words])
    except Exception as e:
        print(e)
        return ""


def create_preprocessing_pipeline(text_data):
    text_prep = Pipeline([("vect", CountVectorizer(min_df=5, ngram_range=(1, 3))),
                          ("tfidf", TfidfTransformer(norm=None))])
    text_prep.fit(text_data)
    return text_prep


def load_and_save_review_data(filepath, review_field, saved_filepath, remove_stopwords=False):
    data = pd.read_csv(filepath, encoding="ISO-8859-1", error_bad_lines=False)
    data["prep_" + review_field] = data[review_field].apply(lambda review: preprocess_review(review, remove_stopwords))
    # data = data.sample(frac=1).reset_index(drop=True)
    joblib.dump(data, saved_filepath)
    return data


def get_cached_filepath(filepath, homepath):
    filename = os.path.split(filepath)[-1][:-4] + "_prep.pkl"
    return os.path.join(homepath, "preprocessed_files", filename)


def preprocess_review_data(filepath, review_field, text_prep=None, homepath=".", cached=True, remove_stopwords=False):
    save_filepath = get_cached_filepath(filepath, homepath)
    if cached:
        if os.path.isfile(save_filepath) and text_prep:
            data = joblib.load(save_filepath)
            return data, text_prep.transform(data["prep_" + review_field]), text_prep

    data = load_and_save_review_data(filepath, review_field, save_filepath, remove_stopwords)
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


def _add_scores(results, prec_rec_f1):
    results["precision"].append(prec_rec_f1[0])
    results["recall"].append(prec_rec_f1[1])
    results["f1"].append(prec_rec_f1[2])


def _write_csv_results(csv_results_filepath, scores):
    categories = []
    f1s = []
    precisions = []
    recalls = []
    classifiers = []
    for category, scores in scores.iteritems():
        categories.append(category)
        f1s.append(scores["f1"])
        precisions.append(scores["precision"])
        recalls.append(scores["recall"])
        classifiers.append(scores["classifier"])
    evaluation_data = pd.DataFrame({
        "category": categories,
        "f1": f1s,
        "precision": precisions,
        "recall": recalls,
        "classifier": classifiers
    })
    evaluation_data.to_csv(csv_results_filepath, index=False)


def evaluate_classification(filepath, results_filepath, csv_results_filepath, review_field, categories, k, clf, predict,
                            homepath):
    print("Writing evaluation scores to: %s" % results_filepath)

    with open(results_filepath, "a+") as writer:
        writer.write(">>>>> Evaluating %s\n" % clf)
        all_scores = {}
        time_1 = time.time()
        data, X, _ = preprocess_review_data(filepath=filepath, review_field=review_field, text_prep=None,
                                            homepath=homepath, cached=False)
        print(data.dtypes)
        print(data.columns)
        diff = time.time() - time_1
        print(">>>>> Preprocessing: %.2f seconds\n" % diff)
        writer.write(">>>>> Preprocessing: %.2f seconds\n" % diff)
        time_1 = time.time()
        for category in categories:
            print("\n\n>>>>> For category: %s\n" % category)
            writer.write("\n\n>>>>> For category: %s\n" % category)

            y = data[category]
            splitter = StratifiedShuffleSplit(n_splits=k, test_size=0.1, random_state=0)

            scores = {
                "f1": [],
                "precision": [],
                "recall": []
            }
            for train_idx, test_idx in splitter.split(X, y):
                X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y.iloc[train_idx], y.iloc[test_idx]
                clf.fit(X_train.toarray(), y_train)
                diff = time.time() - time_1
                writer.write(">>>>> Fitting the classifier: %.2f seconds\n" % diff)
                time_1 = time.time()
                y_pred = predict(clf, X_test.toarray())
                diff = time.time() - time_1
                writer.write(">>>>> Prediction: %.2f seconds\n" % diff)
                time_1 = time.time()
                prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)
                _add_scores(scores, prec_rec_f1)
                print(classification_report(y_test, y_pred))
                print("precision: %.2f recall: %.2f f1_score: %.2f\n" %
                             (prec_rec_f1[0], prec_rec_f1[1], prec_rec_f1[2]))
                writer.write(classification_report(y_test, y_pred))
                writer.write("precision: %.2f recall: %.2f f1_score: %.2f\n" %
                             (prec_rec_f1[0], prec_rec_f1[1], prec_rec_f1[2]))
                diff = time.time() - time_1
                writer.write(">>>>> One evaluation loop: %.2f seconds\n" % diff)
                writer.flush()
                time_1 = time.time()

            for score, values in scores.iteritems():
                scores[score] = sum(values) / len(values)
            scores["classifier"] = str(clf)
            all_scores[category] = scores
            writer.write("For category: %s\n" % category)
            writer.write("Results: %s\n" % scores)
            pkl_file = open(os.path.join(".", "internal_data", "scores.pkl"), 'wb')
            pickle.dump(scores, pkl_file, -1)
        writer.write("\n\nFINAL REPORT FOR %s\n\n" % type(clf))
        print("\n\nFINAL REPORT FOR %s\n\n" % type(clf))
        for category, scores in all_scores.iteritems():
            writer.write("For category: %s\n" % category)
            writer.write("Results: %s\n" % scores)
            print("For category: %s\n" % category)
            print("Results: %s\n" % scores)

        _write_csv_results(csv_results_filepath, all_scores)

    print("Finished writing evaluation scores to %s." % results_filepath)
    return all_scores


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


def train_and_save_model(review_field, category, model_config, models_dir="best_models"):
    data, X, text_prep = preprocess_review_data(filepath=model_config["data_path"], review_field=review_field,
                                                homepath="..", cached=False,
                                                remove_stopwords=model_config["with_stopwords"])
    directory = os.path.join(".", models_dir, category)
    print("Training classifier for category %s" % category)
    clf = train_classifier(ensemble.GradientBoostingClassifier(verbose=2, n_estimators=model_config["n_estimators"]), X,
                           data[category])
    model_details = {"text_field": review_field, "category": category}

    if not os.path.exists(directory):
        os.makedirs(directory)
    joblib.dump(text_prep, os.path.join(directory, "text_prep.pkl"))
    joblib.dump(clf, os.path.join(directory, "model.pkl"))
    joblib.dump(model_details, os.path.join(directory, "model_details.pkl"))
    print("Finished training and saving classifier for category %s" % category)


def build_categories_list():
    categories_lists = cat_subcat_dict.values()
    categories = []
    for category_list in categories_lists:
        categories.extend(category_list)
    return ["IS_" + category for category in categories]


def build_pretty_categories_list():
    return zip(build_categories_list(), [category[3:].lower().title() for category in build_categories_list()])


def build_pretty_categories_list_with_definitions():
    pretty_categories_definitions = []
    for category in build_categories_list():
        category = category[3:]
        pretty_categories_definitions.append((category.lower().title(), categories_definitions[category],
                                              category.lower().replace(" ", "_")))
    return pretty_categories_definitions


def find_category_index(category, categories_list):
    for index, cat_tuple in enumerate(categories_list):
        if category == cat_tuple[0] or category in cat_tuple[0]:
            return index
    return -1


def build_pretty_categories_list_with_checked(filtering_categories):
    pretty_categories = build_pretty_categories_list()
    complaint_index = find_category_index(COMPLAINT, pretty_categories)
    pretty_categories.insert(complaint_index + 1, ('IS_NOT_COMPLAINT', 'Not Complaint'))
    for i in range(len(pretty_categories)):
        pretty_categories[i] += ("checked",) if pretty_categories[i][0] in filtering_categories else ("", )

    return pretty_categories


def preprocess_category_name(category):
    lower_category = category[3:].lower()
    return "_".join(lower_category.split())


def get_reviews_filepath_cached(filepath):
    filename = filepath[filepath.rfind("/") + 1:]
    return os.path.join(".", "classified_reviews_cache", filename[:-4] + ".pkl")


def classify(filepath, review_field, categories, homepath=".", models_dir_name="models_200"):
    text_prep = joblib.load(os.path.join(homepath, "classification", models_dir_name, "text_prep.pkl"))
    data, X, _ = preprocess_review_data(filepath, review_field, text_prep, homepath=homepath)
    pred_categories = []
    for category in categories:
        print("Classifying for category: %s" % category)

        directory = os.path.join(homepath, "classification", models_dir_name, category)
        clf = joblib.load(os.path.join(directory, "model.pkl"))
        y_pred = clf.predict(X.toarray())
        data["PREDICTED_" + category] = [preprocess_category_name(category) if pred == 1 else "" for pred in y_pred]
        pred_categories.append("PREDICTED_" + category)
    data["predictions"] = data[pred_categories].apply(lambda cats: [cat for cat in cats if cat], axis=1)
    return data


def classify_and_save_results(filepath, review_field, categories, cached=True):
    cached_filepath = get_reviews_filepath_cached(filepath)
    if os.path.isfile(cached_filepath) and cached:
        data = joblib.load(cached_filepath)
        return data
    data = classify(filepath, review_field, categories)
    joblib.dump(data, cached_filepath)
    return data