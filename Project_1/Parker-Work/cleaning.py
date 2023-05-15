
#%%

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import altair as alt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
customers = pd.read_csv('bankdata.csv')
customers.head()



#%%
#customers['age'].value_counts() #No wierd types
#customers['marital'].value_counts() #unknows = 69
#customers['job'].value_counts() #unknowns = 294
#customers['education'].value_counts() #unknowns =1535
#customers['default'].value_counts() #unknows = 7725
#customers['housing'].value_counts() #unknows = 894
#customers['loan'].value_counts() #unknows = 894
# customers['y'].value_counts() #unknows = 894

#%%
#Cleaning all the unknowns in the first 7 columns
testClean = customers
testClean["job"] = testClean["job"].replace(['unknown'], "admin.")
testClean["marital"] = testClean["marital"].replace(['unknown'], "married")
testClean["education"] = testClean["education"].replace(['unknown'], "university.degree")
testClean["default"] = testClean["default"].replace(['unknown'], "no")
testClean["housing"] = testClean["housing"].replace(['unknown'], "yes")
testClean["loan"] = testClean["loan"].replace(['unknown'], "no")
testClean.head()
#%%
#putting the data in a clean outcome thanks Jared
testClean['poutcome_clean'] = testClean['poutcome'].apply(lambda x: 0 if x == 'nonexistent' or x == 'failure' else 1)
testClean['pdays_clean'] = testClean['pdays'].apply(lambda x: 0 if x == 999 else x)
filtered_data = testClean[~testClean['pdays_clean'].isna()]


columns_list = list(testClean.columns)
print(columns_list)

# %%

X = testClean.filter(items= ['job', 'education', 'poutcome_clean', 'pdays_clean', 'housing' ])


y = testClean.filter(items= ["y"])

X = pd.get_dummies(X)
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2)


TraingModel = KNeighborsClassifier(n_neighbors= 15)
TraingModel.fit(X_train, y_train)
Y_Predictions = TraingModel.predict(X_test)

# conmat = metrics.confusion_matrix(y_test, Y_Predictions)
#%%
# metrics.ConfusionMatrixDisplay(conmat).plot()
print("Accuracy:", metrics.accuracy_score(y_test, Y_Predictions))
# %%
