# def train_test_models(X,y, cv = 5, metrics = ['accuracy',]):
# Created by benny at 09/11/18
# Code from: https://scikit-learn.org/stable/
# auto_examples/model_selection/plot_confusion_matrix.html
"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""

from .data_preprocessing import check_Xy
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import StandardScaler
from sklearn import metrics



# print(__doc__)
# TODO: Document model selection module

def plot_confusion_matrix(cm, classes=None,
                          normalise=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalise=True`.
    """
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if classes is not None:
        rot = 45
        # TODO: correct problem diff length of labels
    else:
        classes = [x for x in range(1, len(cm) + 1)]
        rot = 0

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)  # #?? An old trick
    tick_marks = np.arange(len(classes)) - .5
    plt.xticks(tick_marks, classes, rotation=rot)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return


def plot_confusion_matrix_grp(conf_mat, **kwargs):
    title_wo_norm = 'Confusion Matrix'
    title_w_norm = 'Normalised CM'
    normalise = kwargs.pop('normalise', False)
    title = title_w_norm if normalise else title_wo_norm

    ncols = kwargs.pop('ncols', 2)
    nrows = kwargs.pop('nrows', math.ceil(len(conf_mat) / ncols))
    classes = kwargs.pop('classes', None)

    index = 1

    # definitely not the most elegant way to get the number of classes
    nclasses = len(list(conf_mat.values())[0])
    pad = 2
    figsize = kwargs.pop('figsize',
                         ((nclasses + pad) * ncols, (nclasses + pad) * nrows))
    plt.figure(figsize=figsize)

    for name, cm in conf_mat.items():
        plt.subplot(nrows, ncols, index)
        plot_confusion_matrix(cm, classes=classes, normalise=normalise, title=title)

        index += 1
    plt.show()


# Code from: https://scikit-learn.org/stable/auto_examples/classification/
# plot_classifier_comparison.html#
# sphx-glr-auto-examples-classification-plot-classifier-comparison-py
# !/usr/bin/python
# -*- coding: utf-8 -*-
# """
# =====================
# Classifier comparison
# =====================
#
# A comparison of a several classifiers in scikit-learn on synthetic datasets.
# The point of this example is to illustrate the nature of decision boundaries
# of different classifiers.
# This should be taken with a grain of salt, as the intuition conveyed by
# these examples does not necessarily carry over to real datasets.
#
# Particularly in high-dimensional spaces, data can more easily be separated
# linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
# might lead to better generalization than is achieved by other classifiers.
#
# The plots show training points in solid colors and testing points
# semi-transparent. The lower right shows the classification accuracy on the test
# set.
# """
# print(__doc__)

# Code source: Gael Varoquaux
#              Andreas Muller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

def compare_classifiers(X, y, **kwargs):
    scoring = kwargs.pop('scoring', 'accuracy')
    cv = kwargs.pop('cv', 5)
    n_jobs = kwargs.pop('n_jobs', -1)
    cm = kwargs.pop('cm', False)
    return_estimator = kwargs.pop('return_estimator', True)

    if 'names' not in kwargs:
        names = ["Logistic Regression",
                 "Gradient Boosting",
                 "Nearest Neighbors",
                 "Linear SVM",
                 "RBF SVM",
                 "Gaussian Process",
                 "Decision Tree",
                 "Random Forest",
                 "MLP Classifier",
                 "AdaBoost",
                 "Naive Bayes",
                 "QDA"
                 ]
        classifiers = [LogisticRegression(n_jobs=n_jobs),
                       GradientBoostingClassifier(n_estimators=100,
                                                  learning_rate=1.0,
                                                  max_depth=1,
                                                  random_state=0,
                                                  ),
                       KNeighborsClassifier(3),
                       SVC(kernel="linear", C=0.025, gamma='auto'),
                       SVC(C=1, gamma='auto'),
                       GaussianProcessClassifier(1.0 * RBF(1.0),
                                                 warm_start=False,
                                                 n_jobs=n_jobs),
                       DecisionTreeClassifier(max_depth=5),
                       RandomForestClassifier(max_depth=5,
                                              n_estimators=10,
                                              max_features=1,
                                              n_jobs=n_jobs),
                       MLPClassifier(alpha=1),
                       AdaBoostClassifier(),
                       GaussianNB(),
                       QuadraticDiscriminantAnalysis()
                       ]
    else:
        names = kwargs.pop('names')
        classifiers = kwargs.pop('classifiers')

    X, y = check_Xy(X, y)

    # Done this way to keep compatibility
    n_splits = cv
    cv = ShuffleSplit(n_splits=n_splits, test_size=(1 / n_splits), random_state=0)
    score = {}
    for name, clf in zip(names, classifiers):
        clf_score = cross_validate(clf, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs,
                                   return_train_score=True,
                                   return_estimator=return_estimator,
                                   **kwargs)
        score[name] = pd.DataFrame(clf_score)

    if cm:
        labels = sorted(set(y.values))
        
        conf_mat = {}
        for name in names:

            clf_predicted = np.array([], dtype=int)
            all_test_index = np.array([], dtype=int)
            for n, (train_index, test_index) in enumerate(cv.split(X)):
                clf = score[name]['estimator'][n]
                X_test = X[test_index]
                this_prediction = clf.predict(X_test)

                # keep all predictions and test index
                clf_predicted = np.append(clf_predicted,
                                          this_prediction)
                all_test_index = np.append(all_test_index, test_index)
            clf_cm = metrics.confusion_matrix(y[all_test_index], clf_predicted,
                                              labels=labels)
            conf_mat[name] = clf_cm

        return pd.concat(score, axis=1), conf_mat

    return pd.concat(score, axis=1)
