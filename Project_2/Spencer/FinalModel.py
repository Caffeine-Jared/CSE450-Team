import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import math
import random
import os

# Get data
df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv')
holdout_target = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini_answers.csv')


# add sqft_product, year, month, and day columns to dataframe
df['sqft_product'] = df['sqft_living'] * df['sqft_lot']
holdout['sqft_product'] = holdout['sqft_living'] * holdout['sqft_lot']


features = ['sqft_living', 'sqft_lot15', 'floors', 'yr_renovated', 'sqft_product', 'condition', 'bathrooms', 'yr_built', 'bedrooms', 'zipcode', 'lat', 'long', 'waterfront', 'sqft_lot', 'sqft_above', 'view', 'grade', 'sqft_living15']

# Split from fetures and price
X = df[features]
y = df['price']
X_holdout = holdout[features]
y_holdout = holdout_target['price']

# XGB Matrix creations for train/val/val2/test/hold
# Train/Val/Test
X_train, X_set, y_train, y_set = train_test_split(X, y, test_size=0.2, random_state=25)
X_val, X_test, y_val, y_test = train_test_split(X_set, y_set, test_size=0.5, random_state=25)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


# Choose Hyper Parameters
reg_params = {'max_depth': 5,
            'learning_rate': 0.12,
            'eval_metric': 'rmse',
            'random_state': 25,
            'tree_method': 'gpu_hist',
            'gpu_id': 0}


# Train Model 
rounds = 300
model = xgb.train(reg_params, dtrain, rounds)
y_pred = model.predict(dval)


# Validate Model
r2 = r2_score(y_val, y_pred)
print(r2)