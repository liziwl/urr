from model import InputForm
from flask import Flask, render_template, request
from compute import compute
from os import listdir
from os.path import isfile, join
from classification.utils import *


PAGE_COUNT = 20

app = Flask(__name__)


def find_files(files_path="./user_reviews_files/"):
    return [f for f in listdir(files_path) if isfile(join(files_path, f))]


@app.route('/vib1', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        result = compute(form.A.data, form.b.data,
                         form.w.data, form.T.data)
    else:
        result = None
    file_choices = find_files()
    return render_template("view.html", form=form, file_choices=file_choices)


saved_data = {
    "data": pd.DataFrame(),
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


def compute_new_data(selected_file):
    global saved_data
    data = classify_and_save_results("./user_reviews_files/" + selected_file, "reviewText", build_categories_list())
    saved_data["selected_file"] = selected_file
    saved_data["data"] = data
    return data[saved_data["page"] * PAGE_COUNT: (saved_data["page"] + 1) * PAGE_COUNT], build_paging_info(data)


def get_paged_data(page):
    global saved_data
    data = pd.DataFrame()
    if not saved_data["data"].empty:
        data = saved_data["data"]
        saved_data["page"] = page
        data = data[page * PAGE_COUNT: (page + 1) * PAGE_COUNT]
        return data, build_paging_info(saved_data["data"], page)
    return data, build_paging_info(data)


@app.route("/reviews", methods=["POST", "GET"])
@app.route('/reviews/<int:page>', methods=['GET', 'POST'])
def classify_reviews(page=1):
    selected_file = request.form.get('file_choice', None)
    if selected_file:
        data, paging_info = compute_new_data(selected_file)
    else:
        data, paging_info = get_paged_data(page)
    return render_template("reviews.html", selected_file=selected_file, paging_info=paging_info,
                           data=data.itertuples(index=False))


if __name__ == '__main__':
    app.run(debug=True)