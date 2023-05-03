url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv'

import pandas as pd

dat = pd.read_csv(url)


holdout = dat.sample(frac=0.1)
dat = dat.drop(holdout.index)

# ==================================== K NEIGHBOR 3====================================================

X = dat[['age','job','marital','education','default','housing','loan','contact','month','day_of_week']]
y = dat['y']


# print(X.head())
X = pd.get_dummies(X)
y = pd.get_dummies(y)
# print(X.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .12)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 87%
def get_accuracy_neigh3(x_train, x_test, y_train, y_test):

    neigh = KNeighborsClassifier(n_neighbors=3)

    neigh.fit(x_train, y_train)

    y_predictions = neigh.predict(x_test)


    print(accuracy_score(y_test, y_predictions, normalize=True, sample_weight=None))





# =================================== k neighbors = 5 ================================

# 88.2
def get_accuracy_neigh5(x_train, x_test, y_train, y_test):

    neigh = KNeighborsClassifier(n_neighbors=5)

    neigh.fit(x_train, y_train)

    y_predictions = neigh.predict(x_test)


    print(accuracy_score(y_test, y_predictions, normalize=True, sample_weight=None))

get_accuracy_neigh5(x_train, x_test, y_train, y_test)



# ================================ Decision tree classifier =============================
from sklearn.tree import DecisionTreeClassifier

# 84%
def get_accuracy_tree(x_train, x_test, y_train, y_test):

    classifier = DecisionTreeClassifier()

    classifier.fit(x_train, y_train)

    y_predictions = classifier.predict(x_test)


    print(accuracy_score(y_test, y_predictions, normalize=True, sample_weight=None))

# get_accuracy_tree(x_train, x_test, y_train, y_test)



# # ====================================depth = 15 ======================== 
# from sklearn.tree import DecisionTreeClassifier

# 87%

def get_accuracy_tree(x_train, x_test, y_train, y_test):

    classifier = DecisionTreeClassifier(max_depth=15)

    classifier.fit(x_train, y_train)

    y_predictions = classifier.predict(x_test)


    print(accuracy_score(y_test, y_predictions, normalize=True, sample_weight=None))

# get_accuracy_tree(x_train, x_test, y_train, y_test)