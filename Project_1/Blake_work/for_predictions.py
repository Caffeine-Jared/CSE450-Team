#%%
import pandas as pd
# from sklearn.utils import resample
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn import over_sampling
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree


# clean = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test.csv'
holdout = pd.read_csv(url)

# # Create some new features
# def create_new_features(data):
#     data['last_contact'] = data['pdays'].apply(lambda x: 1 if x == 999 else 0)
#     data['recent_contact'] = data['pdays'].apply(lambda x: 1 if x < 30 else 0)
#     data['previous_contact'] = data['previous'].apply(lambda x: 1 if x > 0 else 0)

# create_new_features(clean)
# create_new_features(holdout)


# # Encode our features and target as needed
# features = ['nr.employed', 'age', 'euribor3m', "campaign", 'cons.conf.idx', 'poutcome', 'last_contact', 'recent_contact', 'previous_contact']
# X = pd.get_dummies(clean[features], drop_first=True)
# holdout = holdout[features]
# y = clean['y']
# # Split our data into training and test data
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# # Use RandomOverSampler for oversampling
# ros = over_sampling.RandomOverSampler(random_state=42)
# X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# # Define the parameter grid to search over
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Create a random forest classifier
# clf = RandomForestClassifier(random_state=25, n_jobs=-1)

# # Create a GridSearchCV object
# grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)

# # Fit the grid search object to the data
# grid_search.fit(X_train_balanced, y_train_balanced)

# # Print the best hyperparameters and their corresponding accuracy score
# print("Best hyperparameters: ", grid_search.best_params_)
# print("Accuracy score: ", grid_search.best_score_)


# # Build the random forest classifier with the best hyperparameters
# clf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
#                               max_depth=grid_search.best_params_['max_depth'],
#                               min_samples_split=grid_search.best_params_['min_samples_split'],
#                               min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
#                               random_state=25, n_jobs=-1)

# # Train the random forest classifier
# clf.fit(X_train, y_train)

# # Test the random forest classifier
# y_pred = clf.predict(X_test)

# # Generate classification report and confusion matrix
# report = classification_report(y_test, y_pred)
# print(report)

# conmat = confusion_matrix(y_test, y_pred)
# print(conmat)

#%%
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, make_scorer,recall_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# loading in the data
clean = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

# Create new features - apply lambda functions to the pdays and previous columns
def create_new_features(data):
    data['last_contact'] = data['pdays'].apply(lambda x: 1 if x == 999 else 0)
    data['recent_contact'] = data['pdays'].apply(lambda x: 1 if x < 30 else 0)
    data['previous_contact'] = data['previous'].apply(lambda x: 1 if x > 0 else 0)

create_new_features(clean)
create_new_features(holdout)

# encodde features
features = ['nr.employed', 'age', 'euribor3m', "campaign", 'cons.conf.idx', 'poutcome', 'last_contact', 'recent_contact', 'previous_contact']
X = pd.get_dummies(clean[features], drop_first=True)
holdout = pd.get_dummies(holdout[features], drop_first=True)
#%%
print(X.head())
#%%
y = clean['y'].map({'no': 0, 'yes': 1})
# y = holdout['y'].map({'no': 0, 'yes': 1})
#%%
# Train + test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# parameter search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = RandomForestClassifier(random_state=25, n_jobs=-1)

grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring='recall')

grid_search.fit(X_train_balanced, y_train_balanced)

# Get the best estimator
best_model = grid_search.best_estimator_

print("Best hyperparameters: ", grid_search.best_params_)
print("Recall score: ", grid_search.best_score_)

# Use the best model to make predictions
y_pred = best_model.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

conmat = confusion_matrix(y_test, y_pred)
print(conmat)



import numpy as np

# holdout = pd.get_dummies(holdout[features], drop_first=True)


predictions = grid_search.predict(holdout)


predictions = pd.DataFrame(predictions)


# predictions[0] = np.where(predictions[0] == 'no', 0, 1)

predictions.rename(columns = {0:'predictions'}, inplace = True)

print(predictions.head())

predictions.to_csv(path_or_buf=r'C:\Users\Blake Dennett\Downloads\Spring2023\machineLearning\team_04\CSE450-Team\Blake_work\team4-module2-predictions.csv', index=False)


# #%%