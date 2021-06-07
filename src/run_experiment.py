#!/usr/bin/env python3

import os.path
import argparse
# import joblib
import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
# from sklearn.metrics import confusion_matrix

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
    # TODO: read from a csv - not necessary for now
    parser.add_argument('--csv', metavar='.csv', type=file, default=None,
                        help='Dataframe csv file')

    parser.add_argument('--locations', action='store_true',
                        help='Dataframe contains locations labels')

    parser.add_argument('--multiclass', action='store_true',
                        help='Make problem a multiclass one')
    parser.add_argument('--name', metavar='S', type=str, default='exp',
                        help='Experiment name')

    parser.add_argument('--test-size', metavar='0.N', type=float, default=0.2,
                        help='Size of the test split for CV')
    parser.add_argument('--scoring', metavar="{'score1':metric1, ...}",
                        type=json.loads,
                        default={'f1': 'f1_micro', 'AUC': 'roc_auc'},
                        help='Data subset name')

    return parser.parse_args(args)


def main(*args):
    parsed_args = handle_args(*args)
    exp_prefix = parsed_args.name + '_'
    print(f'Running experiment: {parsed_args.name}')

    ########################################################
    # Data import
    ########################################################
    # read in dataset
    if parsed_args.pickle:
        dataset = import_dataset(parsed_args.pickle)
    elif parsed_args.csv:
        dataset = import_dataset(parsed_args.csv, 'csv')
    else:
        raise ValueError("No source")

    # TODO remove after testing
    dataset = dataset.loc['00001']
    # get labels
    if parsed_args.locations:
        dataset, labels = split_labels(dataset, parsed_args.multiclass)
        y = labels

    assert (len(y) == len(dataset))
    print(f'\tdataset size: {len(dataset)}'
          f'\tfeatures: {set(dataset.columns.get_level_values(0))}')

    ########################################################
    # Transform dataset
    ########################################################
    # By defaults transformation includes scale and imputer
    transformation_pipe = transformation_pipeline()
    transformation_pipe.fit(dataset)
    X = transformation_pipe.transform(dataset)

    # Splitting
    # Default: 80% training cross validation, 20% testing best model
    # train_test_split makes an stratified split by default
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=parsed_args.test_size)
    print(f'\tX_train: {len(X_train)}, y_train: {len(y_train)}')

    # Training a Simple Model
    # A quick control test
    print("Running control model")
    model_pipe = model_pipeline()
    model_pipe.fit(X_train, y_train)
    control_predict = model_pipe.predict(X_test)
    control_predict_prob = model_pipe.predict_proba(X_test)
    print(f'\t micro f1-score: {f1_score(y_test, control_predict, average="micro")}'
          f'\t weighted ovo ROC_AUC: '
          f'{roc_auc_score(y_test,control_predict_prob, average="weighted", multi_class="ovo")}')
    print(f'\t classification report:'
          f'{classification_report(y_test, control_predict)}')

    # Training a set of models with diff parameters with GridSearch
    print("Running grid search")
    model_grid = model_GridSearchCV(n_jobs=-1, cv=5,
                                    scoring=parsed_args.scoring,
                                    )
    model_grid.fit(X_train, y_train)
    print(f'\tgrid search scores: {model_grid.score(X_test, y_test)}')
    print("The best parameters are %s with a score of %0.2f"
          % (model_grid.best_params_, model_grid.best_score_))

    # storing results in filesystem
    time = datetime.now().strftime("%H%M%S%d%m%y")
    output_file = exp_prefix + 'cv_results_' + time
    print(f'Results stored as: {output_file}')
    pd.DataFrame(model_grid.cv_results_).to_csv(output_file + '.csv')


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
