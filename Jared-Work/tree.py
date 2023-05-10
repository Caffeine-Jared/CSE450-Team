#%%
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


clean = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

# Create some new features
def create_new_features(data):
    data['last_contact'] = data['pdays'].apply(lambda x: 1 if x == 999 else 0)
    data['recent_contact'] = data['pdays'].apply(lambda x: 1 if x < 30 else 0)
    data['previous_contact'] = data['previous'].apply(lambda x: 1 if x > 0 else 0)

create_new_features(clean)

# Encode our features and target as needed
features = ['nr.employed', 'age', 'euribor3m', "campaign", 'cons.conf.idx', 'poutcome', 'last_contact', 'recent_contact', 'previous_contact']
X = pd.get_dummies(clean[features], drop_first=True)
y = clean['y']
# Split our data into training and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
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
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)

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

# #%%
# # Choose a single decision tree to visualize (e.g., the first tree)
# tree_to_visualize = clf.estimators_[0]

# # Set up the figure
# fig, ax = plt.subplots(figsize=(100, 100))

# # Plot the decision tree
# plot_tree(tree_to_visualize, feature_names=X.columns, class_names=['no', 'yes'], filled=True, rounded=True, ax=ax)

# # Save the plot as a high-resolution image
# fig.savefig("decision_tree.png", dpi=200)  # Set the DPI to a high value for better resolution

# # Show the plot
# plt.show()
# #%%