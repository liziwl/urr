import pandas as pd


def main():
    data = pd.read_csv("./reviews_dataset_no_duplicates.csv", encoding="ISO-8859-1", error_bad_lines=False)
    print(len(data))
    all_apps = set(data["app"])
    for app in all_apps:
        app_data = data.loc[data["app"] == app]
        if len(app_data) > 1000:
            app_data.to_csv("./" + app + ".csv", encoding="ISO-8859-1")


if __name__ == "__main__":
    main()