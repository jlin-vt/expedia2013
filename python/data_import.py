import os
import json
import pandas as pd
from datetime import datetime
from operator import itemgetter
import csv
import pickle

def load_data(train, nrows=5):
    """
    Read data and show its relevant information.

    Args:
        train: data object.
        nrows: the number of rows to read in.

    Returns:
        None.
    """

    print("Print the data basic information:")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("It has {} observations and {} features".format(train.shape[0], train.shape[1]))
    print ("\n")

    print("Print the data completeness: ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(train.isnull().sum()/train.shape[0])
    print ("\n")

    print("Print data type:")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(train.dtypes)
    print ("\n")

    print("Print the first {} rows of data:".format(nrows))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(train.head(nrows))
    print ("\n")

    print("Print the information 1st outcome: ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("{} searchs result in {} clicks".format(train.shape[0], train.loc[train.click_bool==1, ].shape[0]))
    print ("\n")

    print("Print the information 2nd outcome: ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("{} searchs result in {} clicks".format(train.shape[0], train.loc[train.booking_bool==1, ].shape[0]))
    print ("\n")

def get_paths():
    """
    Extract paths from JSON file.

    Args:
        None.

    Returns:
        paths: directory of object.
    """
    paths = json.loads(open("SETTINGS.json").read())
    return paths

def load_train():
    """
    Load training data set.
    """
    print("Reading training data...")
    tstart = datetime.now()
    train_path = get_paths()["train_path"]
    x = pd.read_csv(train_path, nrows=10000)
    print("Time used:" + str(datetime.now() - tstart) + "\n")
    return x

def load_test():
    """
    Load test data set.
    """
    print("Reading test data...")
    tstart = datetime.now()
    test_path = get_paths()["test_path"]
    x = pd.read_csv(test_path, nrows=10000)
    print("Time used:" + str(datetime.now() - tstart) + "\n")
    return x

def save_model(model, isBook=True):
    if isBook:
        out_path = get_paths()["model_path_book"]
    else:
        out_path = get_paths()["model_path_click"]
    pickle.dump(model, open(out_path, "w"))

def load_model(isBook=True):
    if isBook:
        in_path = get_paths()["model_path_book"]
    else:
        in_path = get_paths()["model_path_click"]
    return pickle.load(open(in_path))

def write_submission(recommendations, submission_file=None):
    submission_path = get_paths()["submission_path"]
    rows = [(srch_id, prop_id)
        for srch_id, prop_id, rank_float
        in sorted(recommendations, key=itemgetter(0,2))]

    submission = pd.DataFrame.from_records(rows, columns=["SearchId", "PropertyId"])
    submission.to_csv(submission_path, index=False, sep=',')

def main():
    print('Reading file sizes')
    for f in os.listdir('../data'):
        if 'zip' not in f:
            print(f.ljust(30) + str(round(os.path.getsize('../data/' + f) / 1000000, 2)) + 'MB')
    print ("\n")

    train = load_train()
    test = load_test()
    load_data(train)

if __name__ == "__main__":
    main()
