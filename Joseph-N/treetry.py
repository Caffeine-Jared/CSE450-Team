#%%
#Imports
import pandas as pd
import altair as alt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set_theme(style="ticks")
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

testClean = data
testClean["job"] = testClean["job"].replace(['unknown'], "admin.")
testClean["marital"] = testClean["marital"].replace(['unknown'], "married")
testClean["education"] = testClean["education"].replace(['unknown'], "university.degree")
testClean["default"] = testClean["default"].replace(['unknown'], "no")
testClean["housing"] = testClean["housing"].replace(['unknown'], "yes")
testClean["loan"] = testClean["loan"].replace(['unknown'], "no")
testClean['poutcome'] = testClean['poutcome'].replace(['nonexistent'], "failure")
testClean['pdays'] = testClean['pdays'].apply(lambda x: 0 if x == 999 else x)
testClean[['job','marital',"education",'default','housing','contact','month','day_of_week','poutcome','loan','y']] = testClean[['job','marital',"education",'default','housing','contact','month','day_of_week','poutcome','loan','y']].apply(lambda x: pd.factorize(x)[0])
testClean = testClean[~testClean['pdays'].isna()]
testClean.info()

#%%
features = ['nr.employed', 'age', 'euribor3m', "campaign"]
X = pd.get_dummies(testClean[features], drop_first=True)
y = testClean['y']

# Split our data into training and test data, with 30% reserved for testing
X_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# Build the decision tree
clf = DecisionTreeClassifier(criterion="log_loss")
#%%
# Train it
clf.fit(X_train, y_train)

# Test it 
clf.score(X_val, y_val)
#%%
# predict it
y_pred = clf.predict(X_test)

#Score predictions
clf.score(X_test, y_test)

# Note that this gives us an accuracy score, which may not be the best metric.
# See the SciKit-Learn docs for more ways to assess a model's performance, as
# well as methods for cross validation.
# %%
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(report)
# %%
