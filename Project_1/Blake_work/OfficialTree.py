# Load some test data

#%%
import pandas as pd
clean = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test.csv'
holdout = pd.read_csv(url)
clean.head()



#%%
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

#%%
# Encode our features and target as needed
features = ['nr.employed', 'age', 'euribor3m', "campaign", 'cons.conf.idx', 'poutcome']
X = pd.get_dummies(clean[features], drop_first=True)
y = clean['y']

# Split our data into training and test data, 
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

holdout = pd.get_dummies(holdout[features], drop_first=True)


predictions = clf.predict(holdout)


predictions = pd.DataFrame(predictions)


predictions[0] = np.where(predictions[0] == 'no', 0, 1)

predictions.rename(columns = {0:'predictions'}, inplace = True)

print(predictions.head())


# predictions.to_csv(path_or_buf=r'C:\Users\Blake Dennett\Downloads\Spring2023\machineLearning\team_04\CSE450-Team\Blake_work\team4-module2-predictions.csv', index=False)

# print(type(predictions))

# with open('predictions.csv', 'w') as pred_file:

#     pred_file.writelines(predictions)p


# print(report)
# %%
