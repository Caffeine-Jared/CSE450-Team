#%%
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#%%
url_bank = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv'

bank = pd.read_csv(url_bank)
#bank.replace({'unknown': np.nan, 999: np.nan, 'nonexistent': np.nan}, inplace=True)

#%%
# Encode Age into ranges 
bank['age'] = pd.cut(bank['age'], bins=[0, 25, 40, 64, 120], labels=[0, 1, 2, 3])
bank['age'] = bank['age'].astype('int64')

#bank['age'].value_counts()
#%%
# Encode months into numbers
month_to_number = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

# Replace month labels with numbers in the 'months' column
bank['month'] = bank['month'].replace(month_to_number)
#bank['month'].value_counts()

#%% 
columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'poutcome']

for col in columns_to_encode: 
    encoding , names = pd.factorize(bank[col],sort = True) 
    bank[col] = encoding
    #print(f'{col} -> {names}') 

#%%
#X = bank.filter(['job', 'marital', 'education', 'default', 'housing', 'loan', 'age', 'month', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])
X = bank.drop(['y', 'contact', 'day_of_week', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx'], axis=1, inplace=False)
y = bank['y']

X_train, X_test, y_train, y_test = train_test_split(X, y)

max_acc = 0 
best_i = 0

gnb = RadiusNeighborsClassifier(radius=21)

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

#%% 
accuracy = accuracy_score(y_test, y_pred)
#%%
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
# %%
