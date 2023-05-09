# Load some test data

#%%
import pandas as pd
clean = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
clean.head()

#%%
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

#%%
# Encode our features and target as needed
features = ['nr.employed', 'age', 'euribor3m', "campaign", 'cons.conf.idx', 'poutcome']
X = pd.get_dummies(clean[features], drop_first=True)
y = clean['y']

# Split our data into training and test data, 
# X_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.1)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# The issue here is that you are splitting the dataset twice, both times using the entire dataset (X and y). 
# This means that some data points may appear in more than one set (e.g., training and validation, or validation and test sets). 
# This causes data leakage because the model can indirectly learn from the data that it will later be evaluated on, which leads to overestimating the model's performance.

X_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_set, y_set, test_size=0.2)

# Build the decision tree
clf = DecisionTreeClassifier(criterion="log_loss", random_state=25)
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


print(report)
# %%
conmat = metrics.confusion_matrix(y_test, y_pred)
# %%
print(conmat)
# %%
