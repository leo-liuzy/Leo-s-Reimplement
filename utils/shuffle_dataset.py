import numpy as np
import os
import csv
from tqdm import tqdm

seedidx, SEED = 3, 55
np.random.seed(SEED)

PROJ_DIR = os.getcwd() + "/.."
DATA_DIR = PROJ_DIR + "/datasets"
TASKS = ["ag_news_csv", "amazon_review_full_csv", "dbpedia_csv", "yahoo_answers_csv", "yelp_review_full_csv"]
SUFFIX = "csv"

for task in tqdm(TASKS):
    for split, n_examples in [("train", 115000), ("test", 7600)]:
        filename = f"{DATA_DIR}/{task}/{split}.{SUFFIX}"
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                data.append(row)
        np.random.shuffle(data)
        np.random.shuffle(data)

        with open(f"{DATA_DIR}/{task}{seedidx}/{split}.{SUFFIX}", mode='w') as outfile:
            writer = csv.writer(outfile, delimiter=',', quotechar='"')
            for datum in data[:n_examples]:
                writer.writerow(datum)
