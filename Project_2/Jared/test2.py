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

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get feature importance
feature_importance = grid_search.best_estimator_.feature_importances_

# Create a pandas DataFrame with the feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# Sort the DataFrame by the importance in descending order
importance_df = importance_df.sort_values('Importance', ascending=False)

# Select top-n features
n = 10
top_features = importance_df['Feature'].head(n).tolist()

# Print the top features
print("Top features: ", top_features)

# Select these features from your dataset
X_train_selected = X_train[top_features]
X_val_selected = X_val[top_features]
X_test_selected = X_test[top_features]

# Refit the model on the selected features
grid_search.fit(X_train_selected, y_train)

y_pred = grid_search.predict(X_test_selected)
print(y_pred)
print("Mean Absolute Error: " + str(mean_absolute_error(y_pred, y_test)))
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: " + str(rmse))
r2 = r2_score(y_test, y_pred)
print("R-squared: " + str(r2))
