import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

clean = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

# Separate the majority and minority classes
bank_majority = clean[clean['y'] == 'no']
bank_minority = clean[clean['y'] == 'yes']

# Oversample the minority class
bank_minority_oversampled = resample(bank_minority, replace=True, n_samples=len(bank_majority), random_state=42)

# Combine the majority and minority classes
clean_balanced = pd.concat([bank_majority, bank_minority_oversampled])

# Create some new features
clean_balanced['last_contact'] = clean_balanced['pdays'].apply(lambda x: 1 if x == 999 else 0)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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
