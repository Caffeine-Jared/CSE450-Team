import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the data
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

# Use SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Create a random forest classifier with custom hyperparameters
clf = RandomForestClassifier(n_estimators=100,
                             max_depth=20,
                             min_samples_split=5,
                             min_samples_leaf=2,
                             random_state=25, n_jobs=-1)

# Train the random forest classifier
clf.fit(X_train_balanced, y_train_balanced)

# Test the random forest classifier
y_pred = clf.predict(X_test)

# Generate classification report and confusion matrix
report = classification_report(y_test, y_pred)
print(report)

conmat = confusion_matrix(y_test, y_pred)
print(conmat)
