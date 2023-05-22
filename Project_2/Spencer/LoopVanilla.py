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


# Save Folder and loop count
save_folder = r'C:\Users\dogeb\Documents\GitHub\CSE450-Team\Project_2\New Models'
best_r2_so_far = 0
count = 0
count_all = 0


# Get data
df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv')
holdout_target = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini_answers.csv')


# add sqft_product, year, month, and day columns to dataframe
df['sqft_product'] = df['sqft_living'] * df['sqft_lot']
holdout['sqft_product'] = holdout['sqft_living'] * holdout['sqft_lot']



while True:

    # Choose Features
    sure_features =  [ 'zipcode','lat','long','waterfront','sqft_lot','sqft_above','view','grade','sqft_living15']
    possible_features = ['sqft_living','bedrooms','bathrooms','floors','condition','sqft_basement','yr_built','yr_renovated','sqft_lot15','sqft_product']
    num_features = random.randint(0, len(possible_features) -1)
    features = random.sample(possible_features, num_features) + sure_features 

    # Split from fetures and price
    X = df[features]
    y = df['price']
    X_holdout = holdout[features]
    y_holdout = holdout_target['price']

    # 70/10/10/10 Split
    # Train/Val/Double Check/Test
    X_train, X_set, y_train, y_set = train_test_split(X, y, test_size=0.2, random_state=25)
    X_val, X_set2, y_val, y_set2 = train_test_split(X_set, y_set, test_size=0.33, random_state=25)
    X_val2, X_test, y_val2, y_test = train_test_split(X_set2, y_set2, test_size=0.5, random_state=25)

    # XGB Matrix creations for train/val/val2/test/hold
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dval2 = xgb.DMatrix(X_val2, label=y_val2)
    dtest = xgb.DMatrix(X_test, label=y_test)
    dhold = xgb.DMatrix(X_holdout, label=y_holdout)

    #9,0.1
    # Choose Hyper Parameters
    depth = ['5','6','7','8','9','10']
    rate = ['0.10','0.11','0.12','0.13','0.14','0.15','0.16','0.17','0.18','0.19']
    max_depth = random.sample(depth, 1)
    learning_rate = random.sample(rate, 1)
    max_depth = str(max_depth).strip("'[]'")
    learning_rate = str(learning_rate).strip("'[]'")

    reg_params = {'max_depth': max_depth,
              'learning_rate': learning_rate,
              'eval_metric': 'rmse',
              'random_state': 25,
              'tree_method': 'gpu_hist',
              'gpu_id': 0}


    # Train Model 
    rounds = ['50','100','150','200']
    num_boost_round = random.sample(rounds, 1)
    num_boost_round = str(num_boost_round).strip("'[]'")
    model = xgb.train(reg_params, dtrain, int(num_boost_round))
    y_pred1 = model.predict(dval)
    y_pred2 = model.predict(dval2)

    # Validate Model
    first_r2 = r2_score(y_val, y_pred1)
    second_r2 = r2_score(y_val2, y_pred2)
    avg_val = (first_r2 + second_r2) / 2

    count_all += 1
    print(count_all)

    if avg_val > best_r2_so_far:

        # Document best score
        best_r2_so_far = avg_val
        count += 1

        # Get Test and Holdout scores
        test_predictions = model.predict(dtest)
        holdout_predictions = model.predict(dhold)
        test_r2 = r2_score(y_test, test_predictions)
        h_r2 = r2_score(holdout_predictions, holdout_target)

        #rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        #h_rmse = math.sqrt(mean_squared_error(holdout_predictions, holdout_target))

        # Make text file
        print(f'got one! Number {count}')
        text = 'Score\n'
        text = text + 'val1 r2   = ' + str(first_r2) + '\n'
        text = text + 'val2 r2   = ' + str(second_r2) + '\n'
        text = text + 'avg  r2   = ' + str(avg_val) + '\n'
        text = text + 'test r2   = ' + str(test_r2) + '\n'
        text = text + 'hold r2   = ' + str(h_r2) + '\n\n'
        text = text + 'rounds    = ' + str(num_boost_round) + '\n\n'
        text = text + 'Features\n'

        for f in features:
            text = text + f + '\n'

        file_name = 'avg_r2 = ' + str(format(float(avg_val),".4f")) + '  1_r2 = ' + str(format(float(first_r2),".4f")) + '  2_r2 = ' + str(format(float(second_r2),".4f")) + '.txt'

        # Specify the file name and path
        file_path = os.path.join(save_folder, file_name)

        # Save the text content to a new file
        with open(file_path, 'w') as f:
            f.write(text)
            print (file_name)

        print("New text file created successfully.")
