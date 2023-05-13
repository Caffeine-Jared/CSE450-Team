
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.model_selection import train_test_split


clean = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

# Create some new features
def create_new_features(data):
    data['last_contact'] = data['pdays'].apply(lambda x: 1 if x == 999 else 0)
    data['recent_contact'] = data['pdays'].apply(lambda x: 1 if x < 30 else 0)
    data['previous_contact'] = data['previous'].apply(lambda x: 1 if x > 0 else 0)

create_new_features(clean)

# Encode our features and target as needed
X = pd.get_dummies(clean.drop(['y'], axis=1))
y = clean['y']

mutual_info = mutual_info_classif(X, y)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info.sort_values(ascending=False)

sel_three_feat = SelectKBest(mutual_info_classif, k=9).fit(X, y)
sel_bool = sel_three_feat.get_support()
X_sel = X[X.columns[sel_bool]]

# Split our data into training and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(X_sel, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Use RandomOverSampler for oversampling
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a random forest classifier
clf = RandomForestClassifier(random_state=25, n_jobs=-1)

# Create a GridSearchCV object
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring='precision', verbose=2)

# Fit the grid search object to the data
grid_search.fit(X_train_balanced, y_train_balanced)

# Print the best hyperparameters and their corresponding accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.best_score_)


# Build the random forest classifier with the best hyperparameters
clf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                              max_depth=grid_search.best_params_['max_depth'],
                              min_samples_split=grid_search.best_params_['min_samples_split'],
                              min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                              random_state=25, n_jobs=-1)

# Train the random forest classifier
clf.fit(X_train, y_train)

# Test the random forest classifier
y_pred = clf.predict(X_test)

# Generate classification report and confusion matrix
report = classification_report(y_test, y_pred)
print(report)

conmat = confusion_matrix(y_test, y_pred)
print(conmat)
