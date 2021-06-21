from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# https://stackoverflow.com/questions/50285973/
# pipeline-multiple-classifiers
class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator=SGDClassifier()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


def model_pipeline(steps=None, estimator=SGDClassifier(), **params):
    if steps is None:
        steps = [('clf', ClfSwitcher(estimator=estimator))]
    return Pipeline(steps, **params)


def model_parameters():
    models = {
        'tree':
            {'estimator': DecisionTreeClassifier(),
             'parameters': {'max_depth': [5, 10]}},

        'rf': {'estimator': RandomForestClassifier(),
               'parameters': {'n_estimators': [200, 250],
                              'min_samples_leaf': [1, 5, 10]}},
        'lr': {'estimator': LogisticRegression(penalty='l2'),
               'parameters': {'C': [0.01, 0.1, 1, 10, 100]}},
        'svc':
            {'estimator': SVC(probability=True),
             'parameters': {'kernel': ['linear'],
                            'C': [1, 10]}},
        'svc-rbf': {'estimator': SVC(probability=True),
                    'parameters': {'kernel': ['rbf'],
                                   'gamma': [1e-3, 1e-4],
                                   'C': [1, 10, 100, 1000]}},
    }

    pipeline_parameters = []
    for model, params in models.items():
        model_params = {'clf__estimator': [params.pop('estimator')]}
        for param_name, param_value in params['parameters'].items():
            model_params['clf__estimator__'+param_name] = param_value

        pipeline_parameters.append(model_params)

    print(f'\tGridSearchCV parameters: {pipeline_parameters}\n')
    return pipeline_parameters


def model_GridSearchCV(**kwargs):
    scoring = kwargs.pop('scoring')
    refit = kwargs.pop('refit', None)
    n_splits = kwargs.pop('n_splits', 5)
    n_jobs = kwargs.pop('n_jobs', -1)
    cv = kwargs.pop('cv', None)

    if cv is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False)

    # take the first scoring for refit
    # https://scikit-learn.org/stable/modules/generated/
    # sklearn.model_selection.GridSearchCV.html
    if refit is None:
        refit = next(iter(scoring.keys())) if isinstance(scoring, dict) else False

    pipeline = model_pipeline()
    pipeline_parameters = model_parameters()

    clf_GridSearch = GridSearchCV(pipeline, param_grid=pipeline_parameters,
                                  refit=refit, scoring=scoring,
                                  cv=cv, n_jobs=n_jobs, **kwargs
                                  )
    # TODO: Evaluation by label summary
    return clf_GridSearch
