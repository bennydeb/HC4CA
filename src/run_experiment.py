import os.path
import argparse

from sklearn.model_selection import train_test_split

from HC4CA.data_preprocessing import *
from HC4CA.model_setup import *


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
    parser.add_argument('--name', metavar='S', type=str, default='exp',
                        help='Experiment name')

    return parser.parse_args(args)


def main(*args):
    parsed_args = handle_args(*args)
    exp_prefix = parsed_args.name + '_'
    print(exp_prefix)

    ############################
    # Data import
    ############################
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
        y = labels

    # labels = labels.loc['00001']
    # dataset = dataset.loc['00001']

    ############################
    # Transform dataset
    ############################
    # By defaults transformation includes scale and imputer
    transformation_pipe = transformation_pipeline()
    transformation_pipe.fit(dataset)
    X = transformation_pipe.transform(dataset)

    # Splitting
    # TODO: argument for test_size and move split to data_prepro...
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.5, random_state=0)
    print(f'X_train: {len(X_train)}, y_train: {len(y_train)}')

    # Training a Simple Model
    model_pipe = model_pipeline()
    model_pipe.fit(X_train, y_train)
    print(model_pipe.score(X_test, y_test))

    # Training a set of models with diff parameters with GridSearch
    model_grid = model_GridSearchCV()
    model_grid.fit(X_train, y_train)
    print(model_grid.score(X_test, y_test))

    # storing results in filesystem
    # TODO: store using pickle?
    pd.DataFrame(model_grid.cv_results_).to_csv(exp_prefix + 'results.csv')


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
