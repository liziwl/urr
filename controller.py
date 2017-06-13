from model import InputForm
from flask import Flask, render_template, request
from werkzeug import secure_filename
from classification.classification_utils import *
from utils import *
from flask import redirect, url_for


app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['csv'])


@app.route('/main', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    file_choices = find_files()
    return render_template("view.html", form=form, file_choices=file_choices,
                           categories=build_pretty_categories_list_with_definitions())


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


@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    form = InputForm(request.form)
    error_msg = None
    success_msg = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if allowed_file(app, file.filename):
                filename = secure_filename(file.filename)
                temp_file = os.path.join("/tmp/", filename)
                file.save(temp_file)
                data = pd.read_csv(temp_file, encoding="ISO-8859-1", error_bad_lines=False)
                if REVIEW_FIELD not in data.columns or RATING_FIELD not in data.columns:
                    error_msg = ERROR_MSG_MISSING_FIELDS % (filename, REVIEW_FIELD, RATING_FIELD)
                else:
                    success_msg = SUCCESS_MSG % filename
                    data.to_csv(os.path.join(".", USER_REVIEWS_HOME, filename), encoding="ISO-8859-1", index=False)
            else:
                error_msg = "The input file for the reviews should be a CSV file."
    file_choices = find_files()
    return render_template("view.html", form=form, file_choices=file_choices, invalid_file_error_msg=error_msg,
                           success_msg=success_msg, categories=build_pretty_categories_list_with_definitions())


if __name__ == '__main__':
    app.run(debug=True)