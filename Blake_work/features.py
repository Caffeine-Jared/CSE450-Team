url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv'

import pandas as pd

dat = pd.read_csv(url)


holdout = dat.sample(frac=0.1)
dat = dat.drop(holdout.index)

# ==================================== ====================================================

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