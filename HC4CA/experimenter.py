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


class GenericObject:

    def __str__(self):
        return f"{self.description}"

    def __repr__(self):
        return f"{self.description}"


class Result(GenericObject):
    pass


class Dataset(GenericObject):
    def __init__(self,
                 description,
                 hid,
                 visit,
                 dataset,
                 labels,
                 test_size=None,
                 train_size=None,
                 shuffle=True,
                 random_state=None,
                 ):
        self.description = description
        self.hid = hid
        self.visit = visit
        self.dataset = dataset
        self.labels = labels
        self.test_size = test_size
        self.train_size = train_size
        self.shuffle = shuffle
        self.random_state = random_state

    def get_train_Xy(self):
        return self.X[self.X_train], self.y[self.y_train]

    def get_test_X(self):
        return self.X[self.X_test]

    def get_test_y(self):
        return self.y[self.y_test]

    def train_test_split(self):


class Experiment(GenericObject):
    def __init__(self,
                 description,
                 data: Dataset = None,
                 results: Result = None,
                 ):
        self.description = description
        self.data = data
        self.results = results

    def run(self):
