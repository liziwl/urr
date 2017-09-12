#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from classification.classification_utils import *
from collections import Counter

CATEGORIES = [u'IS_COMPLAINT', u'IS_PRIVACY', u'IS_HARDWARE', u'IS_DEVICE', u'IS_PERFORMANCE', u'IS_BATTERY',
              u'IS_PRICE', u'IS_APP USABILITY', u'IS_ANDROID VERSION', u'IS_UI', u'IS_LICENSING', u'IS_MEMORY',
              u'IS_SECURITY']
APP_DATA_DIR = "./data/app_data/"


def print_category_data(data, category):
    category_data = data.loc[data[category] == 1]
    print("------------------------------------")
    for index, row in category_data.iterrows():
        print(row["reviewText"])
    print("------------------------------------")


category_words = ["donating", "donate", "donation", "$", "bucks", "free",
                  "lollipop", "marshmallow", "nougat", "kitkat",
                  "sd card", "sensor", "accelerometer", "camera", "sensors", "cpu",
                  "permission", "permissions", "privacy", "personal", "private", "track", "noninvasive", "invasive",
                  "memory error", "memory", "out of memory", "ram", "small", "low memory",
                  "kill battery", "drain", "consume", "battery friendly"
                  ]


def print_data_info(filename):
    data = pd.read_csv(filename)
    print("For filename %s total reviews: %d" % (filename, len(data)))
    print(data.columns)
    for category in build_categories_list():
        category_data = data.loc[data[category] == 1]
        print("For category %s : %d" % (category, len(category_data)))


def split_data(data_filepath, apps_dir):
    data = pd.read_csv(data_filepath)
    apps = set(data["app"])
    for app in apps:
        apps_data = data.loc[data["app"] == app]
        print("For app: %s, we have %d reviews" % (app, len(apps_data)))
        app_filename = "_".join(app.split()) + ".csv"
        app_filepath = os.path.join(apps_dir, app_filename)
        apps_data.to_csv(app_filepath, index=False)
        print("Saving app data to: %s" % app_filepath)


def print_app_infos(data_filepath):
    data = pd.read_csv(data_filepath)
    apps = set(data["app"])
    apps_counter = Counter()
    for app in apps:
        app_data = data.loc[data["app"] == app]
        apps_counter[app] = len(app_data)
    print(apps_counter)
    return apps_counter


def main():
    # print_data_info("./data/all_reviews_data.csv")
    # split_data("./data/all_reviews_data.csv", "./data/app_data/")
    print_app_infos("./data/all_reviews_data.csv")


if __name__ == "__main__":
    main()