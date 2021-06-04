import os.path
import argparse
import pandas as pd

from classes import *
from model_setup import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split


def main(*args):
    parsed_args = handle_args(*args)

    # Data import

    # read in dataset
    if parsed_args.pickle:
        dataset = import_dataset(parsed_args.pickle)
    elif parsed_args.csv:
        dataset = import_dataset(parsed_args.csv, 'csv')
    else:
        raise ValueError("No source")

    # get labels
    if parsed_args.locations:
        dataset, labels = split_labels(dataset, parsed_args.multiclass)

    dataset = dataset.loc['00001']
    labels = labels.loc['00001']

    # Transform dataset
    transformation_pipe = transformation_pipeline()
    transformation_pipe.fit(dataset)
    X = transformation_pipe.transform(dataset)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    print(f'X_train: {len(X_train)}, y_train: {len(y_train)}')

    model_pipe = model_pipeline()
    model_pipe.fit(X_train, y_train)
    print(model_pipe.score(X_test, y_test))

    model_grid = model_GridSearchCV()
    model_grid.fit(X_train, y_train)
    # model_grid.score(X_test, y_test)
    # for key, value in model_grid.cv_results_.keys():
    #     print(f'{key}: {value}')
    print(model_grid.score(X_test, y_test))

    print(pd.DataFrame(model_grid.cv_results_).to_csv('results.csv'))


def import_dataset(source, source_type='pkl'):
    """Get Dataset from somewhere and return a df

    :param source: path or location to dataset
    :param source_type: <pkl | csv | >
    :return:
    """
    dataset = []
    if source_type == 'pkl':
        dataset = DataSubset.read_pickle(source)
    elif source_type == 'csv':
        dataset = DataSubset.read_csv(source)
    return dataset


def split_labels(dataset, multiclass=False):
    column = 'Locations'
    labels = dataset[column]
    labels.index = dataset.index

    # drop locations from main df
    dataset.drop(column, axis=1, level=0, inplace=True)

    # only keep annotated rows
    labels = labels[labels.sum(axis=1) != 0.0]

    # keep only instances with annotation
    new_index = dataset.index.intersection(labels.index, sort=None)
    dataset = dataset.loc[new_index]

    # make it multiclass
    if multiclass:
        labels = labels.idxmax(axis=1)

    assert(len(labels) == len(dataset))

    return dataset, labels


def apply_pipeline(pipeline):
    pass


def transformation_pipeline():
    estimators = [('imputer', SimpleImputer()),
                  ('standardise', StandardScaler()),
                  ]
    return Pipeline(estimators)


# def split_dataset()
# Stratified?


# ___________________
# Data Cleaning
# ___________________
def handle_missing_values(dataset):
    pass


# def handle_noisy_values():

# ___________________
# Data Transformation
# -------------------
def feature_scaling(dataset):
    pass


def feature_resampling(dataset):
    pass

# def encode_categorical():


# Data Reduction
# pca

#
#  handle input arguments
#

def file(filename):
    """
    Check if the file specified by filename exists and if
    it is a file (not directory).
    :param filename: path to file
    :return: same filename
    """
    if os.path.isfile(filename):
        return filename
    else:
        raise FileNotFoundError(filename)


def handle_args(*args):
    #
    # Use argparse to handle parsing the command line arguments.
    #   https://docs.python.org/3/library/argparse.html
    #

    parser = argparse.ArgumentParser(description='Prepare dataset data')
    parser.add_argument('--pickle', metavar='.pkl', type=file, default=None,
                        help='Dataframe pickled file')
    parser.add_argument('--csv', metavar='.csv', type=file, default=None,
                        help='Dataframe csv file')

    parser.add_argument('--locations', action='store_true',
                        help='Dataframe contains locations labels')

    parser.add_argument('--multiclass', action='store_true',
                        help='Make problem a multiclass one')
    # parser.add_argument('--bank', metavar='S', type=str, default='natwest',
    #                     help='Bank name')
    # parser.add_argument('--extension', metavar='ext', type=str, default='pdf',
    #                     help='Statements extension: pdf')
    # parser.add_argument('--table_layout', metavar='str1 str2 str3 ...', nargs="+",
    #                     default=['date', 'type', 'description', 'paid in',
    #                              'paid out', 'balance'],
    #                     help='Table columns layout')
    # parser.add_argument('--date_position', metavar='N', type=int, default=0,
    #                     help='Index of date field on each transaction')
    # parser.add_argument('--date_format', metavar='%d%m%y', type=str,
    #                     default='%d %b %Y',
    #                     help='date format found on transactions')
    # parser.add_argument('--reg_exp', metavar='RE', type=str,
    #                     default="((\d{2}\s[A-Z]{3})\s{0,1}(\d{2}\s[A-Z]{3}))\s(\d{0,9})"
    #                             "\s{0,1}(\w[\S\s]*?)(((\d{0,3}){0,}\.(\d{2}))(\s(-)){0,1})",
    #                     help='regular expresion to extract tx')
    # parser.add_argument('--reg_exp_groups', metavar='RE_G', type=list,
    #                     default=[2, 3, 4, 5, 7, 11],
    #                     help='groups from regexp to extract from tx')
    # parser.add_argument('--input_file', metavar='F', type=file, default=None,
    #                     help='File to read in (optional)')
    parser.add_argument('--output-file', metavar='F', type=str, default='output.csv',
                        help='CSV file to write to')

    return parser.parse_args(args)


if __name__ == "__main__":
    # Get data options and return X,y in a csv
    # data_processing --pickle obj.pkl --sensors pir acceleration:rssi video
    #                 --scaler standard  --impute
    import sys

    main(*sys.argv[1:])
