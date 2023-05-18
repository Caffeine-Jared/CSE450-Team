import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import numpy as np

housing = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')

housing['built_after_1976'] = (housing['yr_built'] > 1976).astype(int)

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).miles

locations = {
    "Amazon_HQ": (47.62246, -122.336775),
    "Microsoft": (47.64429, -122.12518),
    "Starbucks": (47.580463, -122.335897),
    "Boeing_Plant": (47.543969, -122.316443)
}

for name, coords in locations.items():
    housing[name + '_distance'] = housing.apply(lambda row: calculate_distance(row['lat'], row['long'], coords[0], coords[1]), axis=1)

column_names = housing.columns.tolist()
print(column_names)

housing['sqft_product'] = housing['sqft_living'] * housing['sqft_lot']
print(housing['sqft_product'].head())
housing['year'] = pd.to_datetime(housing['date']).dt.year
housing['month'] = pd.to_datetime(housing['date']).dt.month
housing['day'] = pd.to_datetime(housing['date']).dt.day

housing = housing.drop('date', axis=1)

# X = housing.drop('price', axis=1)
# y = housing['price']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = XGBRegressor()

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [6, 10, 15],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'subsample': [0.5, 0.7, 1.0],
#     'colsample_bytree': [0.4, 0.7, 1.0]
# }

# # specify number of loops
# num_loops = 5

# best_scores = []
# best_models = []

# for i in range(num_loops):
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=3, scoring='neg_mean_squared_error')
#     grid_search.fit(X_train, y_train)
    
#     # Get the best model from this run and store it
#     best_scores.append(-grid_search.best_score_)  # Flip the sign to make MSE positive
#     best_models.append(grid_search.best_estimator_)

# # Get index of the model with the best score
# best_index = np.argmin(best_scores)  # We use argmin because we want to minimize MSE

# # Select the best model
# best_model_overall = best_models[best_index]

# # Print best model's parameters
# print(best_model_overall.get_params())

# y_pred = best_model_overall.predict(X_test)

# print("Mean Absolute Error: " + str(mean_absolute_error(y_pred, y_test)))
# rmse = sqrt(mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error: " + str(rmse))
# r2 = r2_score(y_test, y_pred)
# print("R-squared: " + str(r2))
