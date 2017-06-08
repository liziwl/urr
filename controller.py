from model import InputForm
from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from compute import compute
from os import listdir
from os.path import isfile, join
from classification.classification_utils import *
from utils import *
from flask import abort, redirect, url_for
from flask import json


PAGE_COUNT = 20
app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['csv'])


def find_files(files_path="./user_reviews_files/"):
    return [f for f in listdir(files_path) if isfile(join(files_path, f))]


@app.route('/main', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    file_choices = find_files()
    return render_template("view.html", form=form, file_choices=file_choices)


saved_data = {
    "data": pd.DataFrame(),
    "all_data": pd.DataFrame(),
    "page": 0,
    "selected_file": ""
}


def build_paging_info(data, page=0):
    if data.empty:
        return {
            "has_prev": False,
            "has_next": False,
        }
    has_prev = page != 0
    prev = page - 1 if page != 0 else None
    has_next = (page + 1) * PAGE_COUNT < len(data)
    next = page + 1 if has_next else None
    return {
        "has_prev": has_prev,
        "prev": prev,
        "has_next": has_next,
        "next": next
    }


def compute_classified_reviews_data(selected_file):
    global saved_data
    data = classify_and_save_results("./user_reviews_files/" + selected_file, "reviewText", build_categories_list())
    saved_data["selected_file"] = selected_file
    saved_data["data"] = data
    saved_data["all_data"] = data
    return data[saved_data["page"] * PAGE_COUNT: (saved_data["page"] + 1) * PAGE_COUNT], build_paging_info(data)


def compute_analysis_data(data, categories, neg_category="IS_ERROR"):
    analysis_data = {}
    for category, pretty_category in categories:
        if category == neg_category:
            continue
        category_data = data.loc[data["PREDICTED_" + category] != ""]
        neg_category_data = category_data.loc[category_data["PREDICTED_" + neg_category] != ""]
        neg_len = len(neg_category_data)
        pos_len = len(category_data) - neg_len
        total = neg_len + pos_len
        neg_percent = 100.0 * neg_len / total if total != 0 else 0
        pos_percent = 100.0 * pos_len / total if total != 0 else 0
        analysis_data[pretty_category] = (round_nr(pos_percent), round_nr(neg_percent), pos_len, neg_len)
    return analysis_data


def generate_analysis_data(selected_file):
    global saved_data
    compute_classified_reviews_data(selected_file)
    return compute_analysis_data(saved_data["all_data"], build_pretty_categories_list(), "IS_ERROR")


def get_paged_data(page):
    global saved_data
    data = pd.DataFrame()
    if not saved_data["data"].empty:
        data = saved_data["data"]
        saved_data["page"] = page
        data = data[page * PAGE_COUNT: (page + 1) * PAGE_COUNT]
        return data, build_paging_info(saved_data["data"], page)
    return data, build_paging_info(data)


def extract_filtering_categories(request):
    if request and request.json and "filtering_categories" in request.json:
        return request.json["filtering_categories"]
    return []


def reset_filtering(filtering_categories):
    return filtering_categories == ["ALL"]


def get_filtered_data(filtering_categories):
    global saved_data
    saved_data["page"] = 0
    data = saved_data["all_data"]
    if reset_filtering(filtering_categories):
        saved_data["data"] = saved_data["all_data"]
    else:
        for category in filtering_categories:
            data = data.loc[data["PREDICTED_" + category] != ""]
        saved_data["data"] = data
    return data[saved_data["page"] * PAGE_COUNT: (saved_data["page"] + 1) * PAGE_COUNT], build_paging_info(data)


@app.route("/analysis", methods=["POST", "GET"])
def analyze_reviews():
    selected_file = request.args["selected_file"]
    analysis_data = generate_analysis_data(selected_file)
    return render_template("analysis.html", selected_file=selected_file, analysis_data=analysis_data,
                           data_is_empty=not bool(analysis_data),
                           review_categories=build_pretty_categories_list_with_checked([]))


@app.route("/reviews", methods=["POST", "GET"])
@app.route('/reviews/<int:page>', methods=["GET", "POST"])
def classify_reviews(page=1):
    selected_file = request.form.get("file_choice", None)
    action = request.form.get("action", None)
    filtering_categories = extract_filtering_categories(request)
    print("Filtering categories: %s" % filtering_categories)
    if selected_file and action == "Classify":
        data, paging_info = compute_classified_reviews_data(selected_file)
    elif selected_file and action == "Analyze":
        return redirect(url_for("analyze_reviews", selected_file=selected_file))
    elif filtering_categories:
        data, paging_info = get_filtered_data(filtering_categories)
    else:
        data, paging_info = get_paged_data(page)
    return render_template("reviews.html", selected_file=selected_file, paging_info=paging_info,
                           data=data.itertuples(index=False), data_is_empty=data.empty,
                           review_categories=build_pretty_categories_list_with_checked(filtering_categories))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    form = InputForm(request.form)
    error_msg = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_file = os.path.join("/tmp/", filename)
                file.save(temp_file)
                data = pd.read_csv(temp_file, encoding="ISO-8859-1", error_bad_lines=False)
                if REVIEW_FIELD not in data.columns:
                    error_msg = "File %s does not have a %s column" % (filename, REVIEW_FIELD)
                else:
                    data.to_csv(os.path.join(".", USER_REVIEWS_HOME, filename), encoding="ISO-8859-1", index=False)
            else:
                error_msg = "The input file for the reviews should be a CSV file."
    file_choices = find_files()
    return render_template("view.html", form=form, file_choices=file_choices,
                           invalid_file_error_msg=error_msg)


if __name__ == '__main__':
    app.run(debug=True)