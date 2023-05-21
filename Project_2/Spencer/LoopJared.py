import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from geopy.distance import geodesic
import math
import random
import os



save_folder = r'C:\Users\dogeb\Documents\GitHub\CSE450-Team\Project_2\Results With Jared'
best_r2_so_far = [0,0,0,0,0]
count = 0
count_all = 0


df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv')
holdout_target = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini_answers.csv')



# add built_after_1976 column to dataframe - hot encode it
df['built_after_1976'] = (df['yr_built'] > 1976).astype(int)
holdout['built_after_1976'] = (holdout['yr_built'] > 1976).astype(int)

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
    df[name + '_distance'] = df.apply(lambda row: calculate_distance(row['lat'], row['long'], coords[0], coords[1]), axis=1)
    holdout[name + '_distance'] = df.apply(lambda row: calculate_distance(row['lat'], row['long'], coords[0], coords[1]), axis=1)

# add sqft_product, year, month, and day columns to dataframe
df['sqft_product'] = df['sqft_living'] * df['sqft_lot']
holdout['sqft_product'] = holdout['sqft_living'] * holdout['sqft_lot']



while True:


    sure_features =  [ 'zipcode','lat','long']
    possible_features = ['sqft_living','bedrooms','bathrooms','floors','condition','sqft_basement','yr_built','yr_renovated','sqft_lot15','sqft_product','waterfront','sqft_lot','sqft_above','view','grade','sqft_living15', 'Amazon_HQ_distance', 'Microsoft_distance', 'Starbucks_distance', 'Boeing_Plant_distance']
    num_features = random.randint(0, len(possible_features) -1)
    features = random.sample(possible_features, num_features) + sure_features 

    X = df[features]
    y = df['price']
    X_holdout = holdout[features]
    y_holdout = holdout_target['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    dhold = xgb.DMatrix(X_holdout, label=y_holdout)

    reg_params = {'max_depth': 9,
              'learning_rate': 0.1,
              'eval_metric': 'rmse',
              'random_state': 25,
              'tree_method': 'gpu_hist',
              'gpu_id': 0}


    num_boost_round = 100
    model = xgb.train(reg_params, dtrain, num_boost_round)
    y_pred = model.predict(dtest)
    r2 = r2_score(y_test, y_pred)

    count_all += 1
    print(count_all)

    for i in [4, 3, 2, 1, 0]:
        if r2 > best_r2_so_far[i]:
            best_r2_so_far[i] = r2
            count += 1

            holdout_predictions = model.predict(dhold)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            h_r2 = r2_score(holdout_predictions, holdout_target)
            h_rmse = math.sqrt(mean_squared_error(holdout_predictions, holdout_target))

            print(f'got one! Number {count}')
            text = 'Score\n'
            text = text + 'r2   = ' + str(r2) + '\n'
            text = text + 'rmse = ' + str(rmse) + '\n'
            text = text + 'Holdout\n'
            text = text + 'r2   = ' + str(h_r2) + '\n'
            text = text + 'rmse = ' + str(h_rmse) + '\n\n'
            text = text + 'Features\n'

            for f in features:
                text = text + f + '\n'

            file_name = 'r2 = ' + str(format(float(r2),".4f")) + ' h_r2 = ' + str(format(float(h_r2),".4f")) + '.txt'

            # Specify the file name and path
            file_path = os.path.join(save_folder, file_name)

            # Save the text content to a new file
            with open(file_path, 'w') as f:
                f.write(text)
                print (file_name)

            print("New text file created successfully.")

            break