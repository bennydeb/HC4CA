# Created by benny at 29/03/2022
# Handle experiments with this package.

"""
===========
Experiments
===========

Experiments are composed of data, models and results.
Data has preprocessing and splitting func associated
Models have the classifiers, parameters, and models.
Results have scores, scoring measures.
"""
import pandas as pd
import os
from typing import TypeVar

from .model_selection import get_default_clfs
from .data_preprocessing import get_Xy, check_Xy, dummy_preprocessor
from .utils.common import get_timestamp

from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np


class GenericObject:
    def __init__(self,
                 description):
        self.description = description

    def __str__(self):
        return f"'{self.description}'"

    def __repr__(self):
        return f'"{self.description}"'


class Results(GenericObject):
    def __init__(self,
                 description,
                 predictions,
                 scores,
                 ):
        super().__init__(description)
        self.description = description
        self.predictions = predictions
        self.scores = scores

    # def __call__(self, *args, **kwargs):
    #     return self.scores

    def __repr__(self):
        return f"{self.description}"

    def __str__(self):
        return self.print_results()

    def print_results(self):
        string = f"{'   Classifier':<30}{' metric':<15}{'score'}\n"
        for clf, metrics in self.scores.items():
            for metric, val in metrics.items():
                string = string + f"{clf:<30}{metric:<15}{val:1.2f}\n"
        return string

    def print_summary(self):
        df = pd.DataFrame(self.scores).T.describe()
        return df


class Dataset(GenericObject):
    def __init__(self,
                 description,
                 hid,
                 visit,
                 dataset,
                 classes=None,
                 test_size=None,
                 train_size=None,
                 shuffle=True,
                 random_state=None,
                 preprocessor=dummy_preprocessor,
                 ):
        super().__init__(description)
        self.description = description
        self.hid = hid
        self.visit = visit
        self.dataset = dataset
        self.classes = classes
        self.test_size = test_size
        self.train_size = train_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.preprocessor = preprocessor

        self.X = None
        self.y = None
        self.train_idx = None
        self.test_idx = None

        self.train_test_split()

    def __len__(self):
        return self.dataset.shape[0]

    def get_train_Xy(self):
        try:
            X_train = self.X[self.train_idx]
        except KeyError:
            X_train = self.X.iloc[self.train_idx]
        return X_train, self.y[self.train_idx]

    def get_test_X(self):
        try:
            X_test = self.X[self.test_idx]
        except KeyError:
            X_test = self.X.iloc[self.test_idx]
        return X_test

    def get_test_y(self):
        return self.y[self.test_idx]

    def get_test_Xy(self):
        return self.get_test_X(), self.get_test_y()

    def get_Xy(self, flatten=True, label_col='label'):
        if self.X is None and self.y is None:
            X, y = get_Xy(self.dataset,
                          flatten=flatten,
                          label_col=label_col)
            X = self.preprocessor(X)
            self.X, self.y = check_Xy(X, y)
        return self.X, self.y

    def train_test_split(self):
        if self.X is None:
            self.get_Xy()
        try:
            idx = self.X.index
        except AttributeError as e:
            print(f"X has no attribute index\n{e}")
            print(self.X)
            idx = np.arange(len(self.X))

        (train_idx, test_idx) = train_test_split(idx,
                                                 test_size=self.test_size,
                                                 train_size=self.train_size,
                                                 random_state=self.random_state,
                                                 shuffle=self.shuffle,
                                                 stratify=None,
                                                 )
        self.train_idx = train_idx
        self.test_idx = test_idx
        return self.train_idx, self.test_idx


class Models(GenericObject):

    def __init__(self,
                 description,
                 **kwargs):
        super().__init__(description)
        self.description = description
        self.models = kwargs.pop("models",
                                 get_default_clfs(**kwargs))

    def __iter__(self):
        for name, clf in self.models.items():
            yield name, clf

    def __str__(self):
        string = "\tClassifiers:\n"
        for i, model in enumerate(self.get_names()):
            string = string + f"{i + 1}: {model}\n"
        return string

    def get_clfs(self):
        return [clf for _, clf in self.models.items()]

    def get_names(self):
        return [name for name, _ in self.models.items()]

    def fit(self, X, y):
        for model in self.get_clfs():
            model.fit(X, y)
        return self.models

    def predict(self, X_test):
        predictions = {}
        for name, model in self:
            # check if fitted
            try:
                predictions[name] = model.predict(X_test)
            except NotFittedError as e:
                print(repr(e))
                exit(1)
        return predictions


data_types = TypeVar('data_types', dict, Dataset)


class Experiment(GenericObject):
    """
    General Experiment holder
    """

    def __init__(self,
                 description,
                 data: data_types = None,
                 models: Models = None,
                 results: Results = None,
                 scoring=('accuracy',),
                 multi_dataset=False,
                 house=None,
                 train_set=None,
                 test_set=None,
                 uuid_prefix=None,
                 ):
        """

        :param description:
        :param data: Dataset or dict {visit: Dataset} format
        :param models:
        :param results:
        :param scoring:
        :param multi_dataset:
        """
        super().__init__(description)
        self.description = description
        self.data = data
        self.models = models
        self.results = results
        self.scoring = scoring
        self.multi_dataset = multi_dataset
        self.house = house
        self.train_set = train_set
        self.test_set = test_set
        self.uuid_prefix = uuid_prefix

    @staticmethod
    def make_score(score, average="micro", **kwargs):
        def score_(y_true, y_pred):
            return score(y_true, y_pred, average=average, **kwargs)

        return score_

    def score(self, y, predicts):
        """

        :param y:
        :param predicts:
        :return:
        """
        scores = {}
        for clf, y_predict in predicts.items():

            scores[clf] = {}
            # TODO: handle when scoring is a dict
            for metric in self.scoring:
                if metric == 'accuracy':
                    metric_func = accuracy_score
                elif metric == 'f1_micro':
                    metric_func = Experiment.make_score(f1_score, average="micro")
                elif metric == 'f1_macro':
                    metric_func = Experiment.make_score(f1_score, average="macro")
                elif callable(metric):
                    metric_func = metric
                else:
                    raise ValueError(f"Cannot score with {metric}")
                scores[clf][metric] = metric_func(y, y_predict)
        return scores

    def get_train_Xy(self):
        if not self.multi_dataset:
            X, y = self.data.get_train_Xy()
        else:
            X, y = self.data[self.train_set].get_Xy()
        return X, y

    def get_test_Xy(self):
        if not self.multi_dataset:
            X, y = self.data.get_test_Xy()
        else:
            X, y = self.data[self.test_set].get_Xy()
        return X, y

    def run(self):
        # fit
        X, y = self.get_train_Xy()
        self.models.fit(X, y)

        # predict
        X_test, y_test = self.get_test_Xy()
        predicts = self.models.predict(X_test)

        # score
        scores = self.score(y_test, predicts)

        if self.results is None:
            self.results = Results(f"Results for {self.description}",
                                   predicts,
                                   scores)

        return self

    def _get_filename_string(self, path=None, info="score", ext=".csv"):
        t_stamp = get_timestamp()
        filename = ""

        if self.uuid_prefix is not None:
            filename = self.uuid_prefix + "_"

        if path is not None:
            filename = os.path.join(path, filename)

        if self.house is None:
            hid = self.data.hid
        else:
            hid = self.house

        return filename + str(hid) + "_" + info + "_" + t_stamp + ext

    def to_csv(self, path=None, w_summary=True):
        # set filename
        filename = self._get_filename_string(path=path)

        df = pd.DataFrame(self.results.scores)
        df.to_csv(filename)
        print(f"{filename} written to disk")

        if w_summary:
            filename = self._get_filename_string(path=path, info="summary")
            summary_df = pd.DataFrame(self.results.scores).T.describe()
            summary_df.to_csv(filename)
            print(f"{filename} written to disk")
