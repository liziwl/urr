from os import listdir
from os.path import isfile, join
from classification.classification_utils import *
from collections import OrderedDict

PAGE_COUNT = 20

ERROR_MSG_MISSING_FIELDS = "The file '%s' does not have the required '%s' and '%s' columns. If the " \
                           "columns exist but have a different name, please rename them before " \
                           "uploading the file."

SUCCESS_MSG = "The file '%s' was uploaded, you can select it from the 'Select reviews file' drop-down menu."

saved_data = {
    "data": pd.DataFrame(),
    "all_data": pd.DataFrame(),
    "page": 0,
    "selected_file": ""
}


def round_nr(nr, digits=2):
    return float(("%." + str(digits) + "f") % nr)


def find_files(files_path="./user_reviews_files/"):
    return [f for f in listdir(files_path) if isfile(join(files_path, f))]


def build_paging_info(data, selected_file, page=0):
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
        "next": next,
        "total_reviews": len(data),
        "start_review": PAGE_COUNT * page + 1,
        "end_review": min(len(data), PAGE_COUNT * (page + 1)),
        "selected_file": selected_file
    }


def compute_classified_reviews_data(selected_file):
    global saved_data
    data = classify_and_save_results("./user_reviews_files/" + selected_file, "reviewText", build_categories_list())
    saved_data["selected_file"] = selected_file
    saved_data["data"] = data
    saved_data["all_data"] = data
    return data[saved_data["page"] * PAGE_COUNT: (saved_data["page"] + 1) * PAGE_COUNT], build_paging_info(data, saved_data["selected_file"])


def compute_analysis_data(data, categories, neg_category="IS_" + COMPLAINT):
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
    analysis_data = compute_analysis_data(saved_data["all_data"], build_pretty_categories_list(), "IS_" + COMPLAINT)
    analysis_data = OrderedDict(sorted(analysis_data.items(), key=lambda t: t[-1][-1] + t[-1][-2], reverse=True))
    return analysis_data


def get_paged_data(page):
    global saved_data
    data = pd.DataFrame()
    if not saved_data["data"].empty:
        data = saved_data["data"]
        saved_data["page"] = page
        data = data[page * PAGE_COUNT: (page + 1) * PAGE_COUNT]
        return data, build_paging_info(saved_data["data"], saved_data["selected_file"], page)
    return data, build_paging_info(data, saved_data["selected_file"])


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
            # special case of Not Complaint
            if "IS_NOT_COMPLAINT" == category:
                data = data.loc[data["PREDICTED_IS_COMPLAINT"] == ""]
            else:
                data = data.loc[data["PREDICTED_" + category] != ""]
        saved_data["data"] = data
    return data[saved_data["page"] * PAGE_COUNT: (saved_data["page"] + 1) * PAGE_COUNT], \
           build_paging_info(data, saved_data["selected_file"])


def allowed_file(app, filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']