import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

housing = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')

#%%

# Drop 'price' from the dataset - ensure that the dataset is not changed, but a new dataset is created
X = housing.drop(['price'], axis=1)

# Create the target variable, 'price' - This is the variable we want to predict
y = housing['price']

#%%
X['date'] = pd.to_datetime(X['date'])
X['year'] = X['date'].dt.year
X['month'] = X['date'].dt.month
X['day'] = X['date'].dt.day
X = X.drop(['date'], axis=1)
#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%

model = XGBRegressor()
model.fit(X_train, y_train)

#%%
predictions = model.predict(X_test)
predictions
#%%
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))
rmse = sqrt(mean_squared_error(y_test, predictions))
print("Root Mean Squared Error: " + str(rmse))
r2 = r2_score(y_test, predictions)
print("R-squared: " + str(r2))