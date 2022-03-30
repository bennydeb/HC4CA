from HC4CA.experimenter import Models, Experiment, Dataset
from sklearn import datasets
import pandas as pd

# import some data to play with
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
class_nmes = iris.target_names

# Simulating a house day dataset with iris
df = pd.DataFrame(X_iris)
df['label'] = y_iris

# create object with data to use on experiments
data = Dataset("Test dataset", 9933, 'A', df, class_nmes, test_size=0.5)
# obtain training and testing sets.
X_train, y_train = data.get_train_Xy()
X_test = data.get_test_X()
y_test = data.get_test_y()

# object to handle all classifiers as one
models = Models("Testing model")
# a quick view to all models
print(models)


# Experiment object to relate models, data and results.
exp = Experiment("Test experiment",
                 models=models,
                 data=data,
                 scoring=['accuracy', 'f1'])

# scores = exp.score(y_test, pred, scoring=['accuracy', 'f1'])

# training, testing and scoring done here
exp.run()

# get results
print(exp.results)

# get means and std
print(exp.results.print_summary())
