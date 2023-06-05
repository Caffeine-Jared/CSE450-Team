import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv')
holdout_target = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini_answers.csv')

drop = ['price', 'date']
X = df.drop(drop, axis=1)
y = df['price']
X_holdout = holdout.drop(['date'], axis=1)
y_holdout = holdout_target['price']

# 80/10/10 Split
# Train/Val/Double Check/Test
X_train, X_set, y_train, y_set = train_test_split(X, y, test_size=0.2, random_state=25)
X_val, X_test, y_val, y_test = train_test_split(X_set, y_set, test_size=0.5, random_state=25)

# Create a Decision Tree Regressor object
tree = DecisionTreeRegressor()

# Train the model
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_val)
r2 = r2_score(y_val, y_pred)
print("Val r2: " + str(r2))

y_pred = tree.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("test r2: " + str(r2))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse)

y_pred = tree.predict(X_holdout)
r2 = r2_score(y_holdout, y_pred)
print("hold r2: " + str(r2))
rmse = np.sqrt(mean_squared_error(y_holdout, y_pred))
print("Hold RMSE:", rmse)


