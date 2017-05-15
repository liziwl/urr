import pandas as pd

data = pd.read_csv("./data/train_reviews_manual.csv")
for t in data.itertuples(index=False):
    print(t)