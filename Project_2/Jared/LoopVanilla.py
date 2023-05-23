#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
#from geopy.distance import geodesic
import math
import random
import os
import seaborn as sns

#%%

save_folder = r'/content/drive/MyDrive/Project_2/'
best_r2_so_far = [.88,.88,.88,.88,.88]
count = 0
count_all = 0

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini.csv')
holdout_target = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test_mini_answers.csv')

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

import matplotlib.pyplot as plt

plt.figure(figsize=(22,8))
plt.scatter(df.index, df["price"], color="red", lw=3)
plt.plot(df.index, df["price"])
plt.title("Prices of homes")
plt.grid(True)
plt.show()

#%%

fig, ax1 = plt.subplots(7, 3, figsize=(20, 25)) # Adjust the number of subplots based on the number of features you want to plot
k = 0
columns = list(df.columns)
for i in range(7):
    for j in range(3):
        if k < len(columns):
            sns.distplot(df[columns[k]], ax=ax1[i][j], color='green')
            ax1[i][j].grid(True)
            k += 1
plt.show()

#%%


#%%

while True:

    # 'zipcode','lat','long','waterfront','sqft_lot','sqft_above','view','grade','sqft_living15'
    sure_features =  [ 'zipcode','lat','long','waterfront','sqft_lot','sqft_above','view','grade','sqft_living15']
    possible_features = ['sqft_living','bedrooms','bathrooms','floors','condition','sqft_basement','yr_built','yr_renovated','sqft_lot15','sqft_product']
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