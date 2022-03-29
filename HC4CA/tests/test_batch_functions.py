
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # import pandas as pd

    # from sklearn import metrics
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from HC4CA.model_selection import (plot_confusion_matrix_grp,
                                       compare_classifiers)

    # import some data to play with
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target
    class_nmes = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X_iris,
                                                        y_iris,
                                                        test_size=0.5,
                                                        random_state=0)
    models = batch_train_clfs(X_train, y_train) 

    predictions == batch_predict
    
    # Compute confusion matrix
    #    cnf_matrix = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix_grp(confm, classes=class_nmes)

