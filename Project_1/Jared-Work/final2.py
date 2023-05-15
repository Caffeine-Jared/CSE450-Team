import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
url_bank = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv'
bank = pd.read_csv(url_bank)

# Preprocess the data
bank['age'] = pd.cut(bank['age'], bins=[0, 25, 40, 64, 120], labels=[0, 1, 2, 3])
bank['month'] = bank['month'].replace({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'poutcome']
for col in columns_to_encode: 
    encoding, names = pd.factorize(bank[col], sort=True) 
    bank[col] = encoding

# Split the data into training, validation, and test sets
X = bank.drop(['y', 'contact', 'day_of_week', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx'], axis=1)
y = bank['y']
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=25)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=True, random_state=28)

# Train a random forest classifier
clf = RandomForestClassifier(random_state=30)
clf.fit(X_train, y_train)

# Evaluate the performance of the model on the validation set
y_pred_val = clf.predict(X_val)
print("Validation Set Performance:")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_val))
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val))

# Evaluate the performance of the model on the test set
y_pred_test = clf.predict(X_test)
print("Test Set Performance:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred_test))
