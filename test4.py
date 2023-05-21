import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import numpy as np
from skopt import BayesSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tune_sklearn import TuneSearchCV
import joblib
# from google.colab import drive

# # # Mount Google Drive
# drive.mount('/content/drive')

# # # output path is ONLY to a file - this can't be used a subdirectory
# output_path = '/content/drive/MyDrive/Project_2/output.txt'

housing = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')

# add built_after_1976 column to dataframe - hot encode it
housing['built_after_1976'] = (housing['yr_built'] > 1976).astype(int)

# calculate distance from each location
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).miles

locations = {
    "Amazon_HQ": (47.62246, -122.336775),
    "Microsoft": (47.64429, -122.12518),
    "Starbucks": (47.580463, -122.335897),
    "Boeing_Plant": (47.543969, -122.316443)
}

# add distance columns to dataframe
for name, coords in locations.items():
    housing[name + '_distance'] = housing.apply(lambda row: calculate_distance(row['lat'], row['long'], coords[0], coords[1]), axis=1)

# add sqft_product, year, month, and day columns to dataframe
housing['sqft_product'] = housing['sqft_living'] * housing['sqft_lot']
housing['year'] = pd.to_datetime(housing['date']).dt.year
housing['month'] = pd.to_datetime(housing['date']).dt.month
housing['day'] = pd.to_datetime(housing['date']).dt.day

# drop date column
housing = housing.drop('date', axis=1)

# Bin sqft_product into different categories
housing['sqft_product_bins'] = pd.qcut(housing['sqft_product'], q=3, labels=['small_properties', 'medium_properties', 'large_properties'])

# hot encode sqft_product_bins
housing = pd.get_dummies(housing, columns=['sqft_product_bins'])

X = housing.drop('price', axis=1)
y = housing['price']

# scale all columns - not sure if this is necessary
scaler = MinMaxScaler()

# add scaled columns to dataframe
for col in X.columns:
    X[col + '_scaled'] = scaler.fit_transform(X[[col]])

X_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_set, y_set, test_size=0.2)

# define model - alpha=1.0 is default
model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor')

# define parameters to search
param_grid = {
    'n_estimators': [50, 100, 200, 250, 300, 500, 1000],
    'max_depth': [6, 10, 15, 20, 25, 30],
    'learning_rate': [0.01, 0.1, 0.3, 0.5],
}

# define target values - these are the values we want to reach (potentially)
target_mae = 10000
target_rmse = 50000
target_r2 = 0.95

# define initial values - these are the values we start with
mae = float('inf')
rmse = float('inf')
r2 = 0

# loop until we reach target values
while mae > target_mae or rmse > target_rmse or r2 < target_r2:
    # define grid search - scoring is negative mean squared error, cv=3 is 3-fold cross validation, n_jobs=-1 is to use all processors
    grid_search = BayesSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train, y_train)

    feature_importance = grid_search.best_estimator_.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    })

    # FIRST SUB-100k features: Top features:  ['grade', 'waterfront', 'sqft_living', 'Amazon_HQ_distance', 'lat', 'view', 'Starbucks_distance', 'Boeing_Plant_distance', 'sqft_living15', 'Microsoft_distance', 'sqft_product', 'sqft_above', 'year', 'yr_built', 'zipcode', 'bathrooms', 'condition', 'long', 'sqft_lot15', 'yr_renovated', 'floors', 'sqft_lot']
    # importance_df - 22 features 
    importance_df = importance_df.sort_values('Importance', ascending=False)
    n = 22
    top_features = importance_df['Feature'].head(n).tolist()
    print("Top features: ", top_features)
    X_train_selected = X_train[top_features]
    X_val_selected = X_val[top_features]
    X_test_selected = X_test[top_features]

    grid_search.fit(X_train_selected, y_train)

#     printing results
    y_pred = grid_search.predict(X_test_selected)
    mae = mean_absolute_error(y_pred, y_test)
    print("Mean Absolute Error: " + str(mae))
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: " + str(rmse))
    r2 = r2_score(y_test, y_pred)
    print("R-squared: " + str(r2))
    print("\n")
    print("n_estimators: " + str(grid_search.best_estimator_.n_estimators))
    print("max_depth: " + str(grid_search.best_estimator_.max_depth))
    print("learning_rate: " + str(grid_search.best_estimator_.learning_rate))
    print("\n")
    
    # write results to file - append to file
    with open('output.txt', 'a') as f:
        print("final results", file=f)
        print("Number of features: ", n, file=f)
        print("Top features: ", top_features, file=f)
        print("Mean Absolute Error: " + str(mae), file=f)
        print("Root Mean Squared Error: " + str(rmse), file=f)
        print("R-squared: " + str(r2), file=f)
        print("\n", file=f)
        print("n_estimators: " + str(grid_search.best_estimator_.n_estimators), file=f)
        print("max_depth: " + str(grid_search.best_estimator_.max_depth), file=f)
        print("learning_rate: " + str(grid_search.best_estimator_.learning_rate), file=f)
        print("\n", file=f)
    if mae <= target_mae and rmse <= target_rmse and r2 >= target_r2:
        break