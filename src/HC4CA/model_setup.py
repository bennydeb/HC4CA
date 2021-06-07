from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def model_pipeline():
    estimators = [('clf', SVC(probability=True)),
                  ]
    return Pipeline(estimators)


def model_GridSearchCV(**kwargs):
    scoring = kwargs.pop('scoring')
    refit = kwargs.pop('refit', None)
    # take the first scoring for refit
    # https://scikit-learn.org/stable/modules/generated/
    # sklearn.model_selection.GridSearchCV.html
    if refit is None:
        refit = next(iter(scoring.values())) if isinstance(scoring, dict) else False

    pipeline = model_pipeline()
    models = {  # 'rf': {'model': RandomForestClassifier(),
        #        'parameters': {'n_estimators': [200, 250],
        #                       'min_samples_leaf': [1, 5, 10]}},
        # 'lr': {'model': LogisticRegression(penalty='l2'),
        #        'parameters': {'C': [0.01, 0.1, 1, 10, 100]}},
        'svc': {'model': SVC(probability=True),
                'parameters': {'kernel': ['linear'],
                               'C': [1, 10]}},
        # 'svc-rbf': {'model': SVC(probability=True),
        #             'parameters': {'kernel': ['rbf'],
        #                            'gamma': [1e-3, 1e-4],
        #                            'C': [1, 10, 100, 1000]}},
    }

    pipeline_parameters = {'clf__' + key: value for key, value in
                           models['svc']['parameters'].items()}
    clf_GridSearch = GridSearchCV(pipeline, param_grid=pipeline_parameters,
                                  refit=refit, verbose=10, scoring=scoring,
                                  **kwargs
                                  )
    # TODO: Evaluation by label summary
    return clf_GridSearch
