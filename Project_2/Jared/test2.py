import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import numpy as np
from itertools import combinations

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


housing['sqft_product'] = housing['sqft_living'] * housing['sqft_lot']
housing['year'] = pd.to_datetime(housing['date']).dt.year
housing['month'] = pd.to_datetime(housing['date']).dt.month
housing['day'] = pd.to_datetime(housing['date']).dt.day

housing = housing.drop('date', axis=1)

X = housing.drop('price', axis=1)
y = housing['price']

X_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_set, y_set, test_size=0.2)

model = XGBRegressor(alpha=1.0)

param_grid = {
    'n_estimators': [50, 100, 200, 250, 300],
    'max_depth': [6, 10, 15, 20, 25, 30],
    'learning_rate': [0.01, 0.1, 0.3],
}

# specify number of loops
num_loops = 1

best_scores = []
best_models = []
best_features = []

column_names = X.columns.tolist()

# Loop through all lengths of combinations
for r in range(1, len(column_names) + 1):
    for cols in combinations(column_names, r):
        for i in range(num_loops):
            X_train_temp = X_train[list(cols)]
            X_val_temp = X_val[list(cols)]

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_temp, y_train)

            # store best model and score
            best_scores.append(-grid_search.best_score_)
            best_models.append(grid_search.best_estimator_)
            best_features.append(cols)  # Store the used features
            print("Loop: ", i)
            print("Best score: ", -grid_search.best_score_)
            print("Best model: ", grid_search.best_estimator_)
            print("Best features: ", cols)
            print("")
            # Print results so far
            #$sdfgsdfgsdfgsdfgsdfgsdrfgsdfgsdfgsdfgsdfgssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
                  # Print results so far
            #$sdfgsdfgsdfgsdfgsdfgsdrfgsdfgsdfgsdfgsdfgssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
                  # Print results so far
            #$sdfgsdfgsdfgsdfgsdfgsdrfgsdfgsdfgsdfgsdfgssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
                  # Print results so far
            #$sdfgsdfgsdfgsdfgsdfgsdrfgsdfgsdfgsdfgsdfgssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
                  # Print results so far
            #$sdfgsdfgsdfgsdfgsdfgsdrfgsdfgsdfgsdfgsdfgssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
                  # Print results so far
            #$sdfgsdfgsdfgsdfgsdfgsdrfgsdfgsdfgsdfgsdfgssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
                  # Print results so far
            #$sdfgsdfgsdfgsdfgsdfgsdrfgsdfgsdfgsdfgsdfgssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
            print("Results so far: ")
            print("Best scores: ", best_scores)
            print("Best models: ", best_models)
            print("Best features: ", best_features)
            with open('output.txt', 'a') as f:
                print("Loop: ", i, file=f)
                print("Best score: ", -grid_search.best_score_, file=f)
                print("Best model: ", grid_search.best_estimator_, file=f)
                print("Best features: ", cols, file=f)
                print("", file=f)
                print("Results so far: ", file=f)
                print("Best scores: ", best_scores, file=f)
                print("Best models: ", best_models, file=f)
                print("Best features: ", best_features, file=f)

# Get index of the model with the best score
best_index = np.argmin(best_scores)

# Select the best model and best features
best_model_overall = best_models[best_index]
best_features_overall = best_features[best_index]

# Print best model's parameters
print(best_model_overall.get_params())
print("Best features: ", best_features_overall)

X_test_temp = X_test[list(best_features_overall)]
y_pred = best_model_overall.predict(X_test_temp)
print(y_pred)
print("Mean Absolute Error: " + str(mean_absolute_error(y_pred, y_test)))
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: " + str(rmse))
r2 = r2_score(y_test, y_pred)
print("R-squared: " + str(r2))