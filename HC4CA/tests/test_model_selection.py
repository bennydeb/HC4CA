if __name__ == "__main__":
    # import itertools
    # import math
    # import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib inline

    # import pandas as pd

    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler

    # from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier,
                                  AdaBoostClassifier,
                                  # GradientBoostingClassifier,
                                  )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # from sklearn.model_selection import TimeSeriesSplit
    # from sklearn.model_selection import cross_validate
    # from sklearn.model_selection import cross_val_predict
    # from sklearn.model_selection import ShuffleSplit

    # from sklearn import metrics
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from HC4CA.model_selection import (plot_confusion_matrix_grp,
                                       compare_classifiers)

    # import some data to play with
    iris = datasets.load_iris()
    X_iris = iris.data
    # print('one ', X_iris)
    # X_iris = StandardScaler().fit_transform(X_iris)
    # print('two ', X_iris)
    y_iris = iris.target
    class_nmes = iris.target_names

    clf_names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA"
    ]
    clf_classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(C=1),
        #         GaussianProcessClassifier(1.0 * RBF(1.0),
        #                                   max_iter_predict=1000,
        #                                   warm_start=True),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X_iris,
                                                        y_iris,
                                                        test_size=0.5,
                                                        random_state=0)

    scores, confm = compare_classifiers(X_train, y_train,
                                        names=clf_names, classifiers=clf_classifiers,
                                        n_jobs=-1,
                                        cm=True)
    #     print(scores)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    #    classifier = svm.SVC(kernel='linear', C=0.01)
    #    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    # Compute confusion matrix
    #    cnf_matrix = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix_grp(confm, classes=class_nmes)

    # clf_name, cnf_matrix = confm.popitem()
    # np.set_printoptions(precision=2)
    #
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title=f'{clf_name} - Confusion matrix, without normalization')
    #
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalise=True,
    #                       title=f'{clf_name} - Normalized confusion matrix')

    plt.show()
