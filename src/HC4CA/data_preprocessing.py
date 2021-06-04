from .classes import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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


