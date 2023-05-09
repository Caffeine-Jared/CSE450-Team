#%%
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


c_url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv'
clean = pd.read_csv(c_url)
clean.head()
h_url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test.csv'
holdout = pd.read_csv(h_url)
holdout.head()



#%%
# Encode our features and target as needed
features = ['nr.employed', 'age', 'euribor3m', "campaign", 'cons.conf.idx', 'poutcome']
X = pd.get_dummies(clean[features], drop_first=True)
y = clean['y']

# Split our data into training and test data, with 30% reserved for testing
# X_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.1)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)



# First, split the data into a training set and a temporary set using an 80-20 split.
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then, split the temporary set into a validation set and a final test set using a 50-50 split.
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Build the decision tree
clf = DecisionTreeClassifier(criterion="gini")
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

conmat = metrics.confusion_matrix(y_test, y_pred)
print(conmat)


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(confusion_matrix=conmat, display_labels=clf.classes_)
disp.plot()

plt.show()


# recall = 53.29%
# precision = 85.88%



# ======================================== setting up the predictions file ======================================
import numpy as np
holdout = pd.get_dummies(holdout[features], drop_first=True)


predictions = clf.predict(holdout)


predictions = pd.DataFrame(predictions)


predictions[0] = np.where(predictions[0] == 'no', 0, 1)

predictions.rename(columns = {0:'predictions'}, inplace = True)
# %%
