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

# encodde features
features = ['nr.employed', 'age', 'euribor3m', "campaign", 'cons.conf.idx', 'poutcome', 'last_contact', 'recent_contact', 'previous_contact']
X = pd.get_dummies(clean[features], drop_first=True)
#%%
print(X.head())
#%%
y = clean['y'].map({'no': 0, 'yes': 1})
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