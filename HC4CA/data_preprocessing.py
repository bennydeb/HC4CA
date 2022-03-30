from .classes import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from numpy import array


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

    assert (len(labels) == len(dataset))

    return dataset, labels


def apply_pipeline(pipeline):
    pass


def transformation_pipeline(steps=None, **kwargs):
    if steps is None:
        steps = [('imputer', SimpleImputer()),
                 ('standardise', StandardScaler()),
                 ]
    return Pipeline(steps, **kwargs)


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

# General
# from 4Houses_TechnicianVisits.

def get_Xy(df_rssi, flatten=True, label_col='label'):
    y = df_rssi[label_col]
    X = df_rssi.drop(label_col, axis=1)
    if flatten:
        X = X.reset_index(drop=True)
    return X, y


def check_Xy(X, y, as_array=False):
    if as_array:
        X = array(X)
    if len(X) != len(y):
        raise ValueError(f"X has length{len(X)}, and y: {len(y)}")

    return X, y


# RSSI preprocessing
def preprocess_X_rssi(X, value=-120.0):
    """ Deals with NaN, missing values, etc.

    :param X:
    :param value: default:-120.0 db.
    :return:
    """
    return X.fillna(value)


def dummy_preprocessor(X, **kwargs):
    return X
