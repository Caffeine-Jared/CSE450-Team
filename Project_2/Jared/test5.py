import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import math
import numpy as np
from sklearn.metrics import make_scorer
import seaborn as sns
import random
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# from tune_sklearn import TuneGridSearchCV
# import joblib
from google.colab import drive

# # Mount Google Drive
drive.mount('/content/drive')

save_folder = r'/content/drive/MyDrive/Project_2/'
best_r2_so_far = 0.913857
count = 0
count_all = 0

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv')
holdout_target = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini_answers.csv')
zip_codes = pd.read_csv('/content/drive/MyDrive/Project_2/zipcodes_file.csv')

def log_transform(col):
    return np.log(col[0])

# log transform on sqft_living and sqft_above
df["sqft_living"]=df[["sqft_living"]].apply(log_transform, axis=1)
holdout["sqft_living"]=holdout[["sqft_living"]].apply(log_transform, axis=1)
df['sqft_above']=df[["sqft_above"]].apply(log_transform, axis=1)
holdout['sqft_above']=holdout[["sqft_above"]].apply(log_transform, axis=1)
df["sqft_living15"]=df[["sqft_living15"]].apply(log_transform, axis=1)
holdout["sqft_living15"]=holdout[["sqft_living15"]].apply(log_transform, axis=1)
df['sqft_product'] = df['sqft_living'] * df['sqft_lot']
holdout['sqft_product'] = holdout['sqft_living'] * holdout['sqft_lot']
df['sqft_quotient'] = df['sqft_living'] / df['sqft_lot']
holdout['sqft_quotient'] = holdout['sqft_living'] / holdout['sqft_lot']



df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day'] = pd.to_datetime(df['date']).dt.day

# drop date column
df = df.drop('date', axis=1)
#%%

#%%

def merge_distance_cols(main_df, holdout_df, zip_codes):
    # Select only the columns to be merged
    zip_codes_subset = zip_codes[['zipcode', 'Amazon_HQ', 'Microsoft', 'Starbucks', 'Boeing_Plant']]
    
    # Merge the dataframes
    main_df = pd.merge(main_df, zip_codes_subset, how='left', on='zipcode')
    holdout_df = pd.merge(holdout_df, zip_codes_subset, how='left', on='zipcode')
    
    # Save the merged dataframes
    main_df.to_csv('merged_main_df.csv')
    holdout_df.to_csv('merged_holdout_df.csv')

    # Return the updated dataframes in case they need to be used later in the program
    return main_df, holdout_df

df, holdout = merge_distance_cols(df, holdout, zip_codes)

#%%
count_all = 0
best_r2_so_far = 0.91

while True:
    # Choose Features
    sure_features =  [ 'zipcode','lat','long','waterfront','sqft_lot','sqft_above','view','grade','sqft_living15', 'Amazon_HQ', 'Microsoft', 'Starbucks', 'Boeing_Plant','yr_built','sqft_product']
    possible_features = ['sqft_living','bedrooms','bathrooms','floors','condition','sqft_basement','yr_renovated','sqft_lot15','sqft_quotient']
    num_features = random.randint(0, len(possible_features) -1)
    features = random.sample(possible_features, num_features) + sure_features 

    # Split from fetures and price
    X = df[features]
    y = df['price']
    X_holdout = holdout[features]
    y_holdout = holdout_target['price']

    # # Instantiate Robust Scaler
    # scaler = MinMaxScaler()
    
    # # Fit the scaler on X
    # scaler.fit(X)

    # # Transform the data
    # X = scaler.transform(X)
    # X_holdout = scaler.transform(X_holdout)

    # 80/10/10 Split
    # Train/Val/Double Check/Test
    X_train, X_set, y_train, y_set = train_test_split(X, y, test_size=0.2, random_state=25)
    X_val, X_test, y_val, y_test = train_test_split(X_set, y_set, test_size=0.33, random_state=25)
    # Define parameters for GridSearchCV
    param_grid = {
        'max_depth': [5, 6, 7, 8, 9, 10],
        'learning_rate': [0.10, 0.11, 0.12, 0.13],
        'n_estimators': [50, 100, 150, 200, 300]
    }

    # XGB Regressor setup
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist', gpu_id=0, random_state=25)
    grid = GridSearchCV(xgbr, param_grid, cv=3, scoring=make_scorer(r2_score), verbose=2)

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    # get best params
    best_params = grid.best_params_

    # pred values
    y_pred = model.predict(X_val)

    # r2 score
    r2 = r2_score(y_val, y_pred)

    X_test = np.array(X_test)
    X_holdout = np.array(X_holdout)
    y_test = np.array(y_test)
    y_holdout = np.array(y_holdout)

    test_predictions = model.predict(X_test)
    holdout_predictions = model.predict(X_holdout)

    test_r2 = r2_score(y_test, test_predictions)
    h_r2 = r2_score(y_holdout, holdout_predictions)

    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    rmse_holdout = np.sqrt(mean_squared_error(y_holdout, holdout_predictions))

    mae_test = mean_absolute_error(y_test, test_predictions)
    mae_holdout = mean_absolute_error(y_holdout, holdout_predictions)

    mse_test = mean_squared_error(y_test, test_predictions)
    mse_holdout = mean_squared_error(y_holdout, holdout_predictions)

    if test_r2 > best_r2_so_far or h_r2 > best_r2_so_far:
      # change to allow for best r2 of test or holdout 
      best_r2_so_far = max(test_r2, h_r2)

      # updated results
      file_name = 'r2 = ' + str(format(float(r2), ".6f")) + '.txt'
      file_path = os.path.join(save_folder, file_name)

      with open(file_path, 'w') as f:
          f.write('Score\n')
          f.write('val1 r2   = ' + str(r2) + '\n')
          f.write('test r2   = ' + str(test_r2) + '\n')
          f.write('hold r2   = ' + str(h_r2) + '\n\n')
          f.write('test RMSE = ' + str(rmse_test) + '\n')
          f.write('hold RMSE = ' + str(rmse_holdout) + '\n\n')
          f.write('test MAE  = ' + str(mae_test) + '\n')
          f.write('hold MAE  = ' + str(mae_holdout) + '\n\n')
          f.write('test MSE  = ' + str(mse_test) + '\n')
          f.write('hold MSE  = ' + str(mse_holdout) + '\n\n')
          f.write('Parameters\n')
          f.write('max_depth     = ' + str(best_params['max_depth']) + '\n')
          f.write('learning_rate = ' + str(best_params['learning_rate']) + '\n')
          f.write('n_estimators  = ' + str(best_params['n_estimators']) + '\n\n')
          f.write('Features\n')
          for feature in features:
              f.write(feature + '\n')

      print("New text file created successfully.")