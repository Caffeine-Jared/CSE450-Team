url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv'

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import altair as alt

dat = pd.read_csv(url)


# holdout = dat.sample(frac=0.1)
# dat = dat.drop(holdout.index)

# ==================================== ====================================================

X = dat[['age','nr.employed','euribor3m']]
y = dat['y']


X = pd.get_dummies(X)
y = pd.get_dummies(y)

# print(y.head())

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=25)


def get_precision(x_train, x_test, y_train, y_test):

    neigh = KNeighborsClassifier(n_neighbors=5)

    neigh.fit(x_train, y_train)

    y_predictions = neigh.predict(x_test)

    print(y_predictions)

    matrix = metrics.confusion_matrix(y_test.values.argmax(axis=1), y_predictions.argmax(axis=1))

    true_positives = matrix[0][0]
    false_positives = matrix[0][1]
    false_negative = matrix[1][0]

    print(true_positives / (true_positives + false_positives))




get_precision(x_train, x_test, y_train, y_test)


19.94

29.83


84.21
59.26


# =================================== features chart ======================================

