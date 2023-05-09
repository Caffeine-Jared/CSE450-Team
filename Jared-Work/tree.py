#%%
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

clean = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
#%%
# Separate the majority and minority classes
bank_majority = clean[clean['y'] == 'no']
bank_minority = clean[clean['y'] == 'yes']

# Oversample the minority class
bank_minority_oversampled = resample(bank_minority, replace=True, n_samples=len(bank_majority), random_state=42)

# Combine the majority and minority classes
clean_balanced = pd.concat([bank_majority, bank_minority_oversampled])

# Create some new features
clean_balanced['last_contact'] = clean_balanced['pdays'].apply(lambda x: 0 if x == 999 else 0)
clean_balanced['recent_contact'] = clean_balanced['pdays'].apply(lambda x: 1 if x < 30 else 0)
clean_balanced['previous_contact'] = clean_balanced['previous'].apply(lambda x: 1 if x > 0 else 0)

# Feature Engineering
clean_balanced['age_group'] = pd.cut(clean_balanced['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=['0-25', '26-35', '36-45', '46-55', '56-65', '66+'])
clean_balanced['education'] = clean_balanced['education'].replace({"unknown": "unknown_education"})
clean_balanced['job'] = clean_balanced['job'].replace({"unknown": "unknown_job"})
clean_balanced['poutcome_success'] = clean_balanced['poutcome'].apply(lambda x: 1 if x == 'success' else 0)

# Encode our features and target as needed
features = ['nr.employed', 'age', 'euribor3m', "campaign", 'cons.conf.idx', 'poutcome', 'last_contact', 'recent_contact', 'previous_contact']
X = pd.get_dummies(clean_balanced[features], drop_first=True)
y = clean_balanced['y']

# Split our data into training and test data

X_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_set, y_set, test_size=0.2)

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a random forest classifier
clf = RandomForestClassifier(random_state=25, n_jobs=-1)

# Create a GridSearchCV object
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)

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

# Load the holdout dataset
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test.csv')

# Apply the same preprocessing steps as the training dataset
holdout['last_contact'] = holdout['pdays'].apply(lambda x: 0 if x == 999 else 0)
holdout['recent_contact'] = holdout['pdays'].apply(lambda x: 1 if x < 30 else 0)
holdout['previous_contact'] = holdout['previous'].apply(lambda x: 1 if x > 0 else 0)
holdout['age_group'] = pd.cut(holdout['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=['0-25', '26-35', '36-45', '46-55', '56-65', '66+'])
holdout['education'] = holdout['education'].replace({"unknown": "unknown_education"})
holdout['job'] = holdout['job'].replace({"unknown": "unknown_job"})
holdout['poutcome_success'] = holdout['poutcome'].apply(lambda x: 1 if x == 'success' else 0)

# Encode the features in the holdout dataset
X_holdout = pd.get_dummies(holdout[features], drop_first=True)

# Use the trained classifier to make predictions on the holdout dataset
y_holdout_pred = clf.predict(X_holdout)

# Count the occurrences of each class in the predictions
yes_count = (y_holdout_pred == 'yes').sum()
no_count = (y_holdout_pred == 'no').sum()

# Calculate the percentage of 'yes' and 'no' predictions
yes_percentage = (yes_count / len(y_holdout_pred)) * 100
no_percentage = (no_count / len(y_holdout_pred)) * 100

# Print the percentage of 'yes' and 'no' predictions
print(f"Percentage of 'yes' predictions: {yes_percentage:.2f}%")
print(f"Percentage of 'no' predictions: {no_percentage:.2f}%")

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
# %%
