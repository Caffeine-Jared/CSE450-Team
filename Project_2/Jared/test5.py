#%%
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor, plot_tree, plot_importance, to_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
import neptune
from neptune.integrations.xgboost import NeptuneCallback
from matplotlib import pyplot as plt
import seaborn as sns

# # ask the user for the api key
# api_key = input("Please enter your Neptune API key: ")

# # neptune ai implementation
# run = neptune.init_run(
#     project="jaredlin70/CSE450-Project2",
#     api_token=api_key,
# )

# load in data for housing from Seattle
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

# create X and y - drop price from X
X = housing.drop('price', axis=1)
y = housing['price']


#%%

fig, ax1 = plt.subplots(7,2, figsize=(20,25))
k = 0
columns = list(housing.columns)
for i in range(7):
    for j in range(2):
            sns.distplot(housing[columns[k]], ax = ax1[i][j], color = 'green')
            ax1[i][j].grid(True)
            k += 1
plt.show()

#%%

# we probably want to use a robust scaler that isn't effected by outliers - this would be the best way to scale the data, but it might not be necessary if we just use scaler
fig, ax1 = plt.subplots(7,2, figsize=(20,25))
k = 0
columns = list(housing.columns)
for i in range(7):
    for j in range(2):
            sns.distplot(housing[columns[k]], ax = ax1[i][j], color = 'green')
            ax1[i][j].grid(True)
            k += 1
plt.show()

#%%

def log_transform(col):
    return np.log(col[0])

housing["sqft_living"]=housing[["sqft_living"]].apply(log_transform, axis=1)
#Plot
sns.distplot(housing["sqft_living"], color = 'green')
plt.grid(True)
plt.show()
#%%

housing["sqft_above"]=housing[["sqft_above"]].apply(log_transform, axis=1)
#Plot
sns.distplot(housing["sqft_above"], color = 'green')
plt.grid(True)
plt.show()
#%%

plt.figure(figsize=(20,10))
corr=abs(housing.corr())
sns.heatmap(corr,annot=True,linewidth=1,cmap="Blues")
plt.show()

plt.figure(figsize=(20,10))
plt.plot(corr["price"].sort_values(ascending=False)[1:],label="Correlation",color="red")
plt.ylabel("Correlation")
plt.xlabel("Feature")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
#%%

fig, ax = plt.subplots(11, 2, figsize=(20, 50))  # Adjust to create 22 subplots
columns = ['grade', 'waterfront', 'sqft_living', 'Amazon_HQ_distance', 'lat', 'view', 'Starbucks_distance', 'Boeing_Plant_distance', 'sqft_living15', 'Microsoft_distance', 'sqft_product', 'sqft_above', 'year', 'yr_built', 'zipcode', 'bathrooms', 'condition', 'long', 'sqft_lot15', 'yr_renovated', 'floors', 'sqft_lot']
k = 0
for i in range(11):  # Loop over 11 rows
    for j in range(2):  # Loop over 2 columns
        if k < len(columns):  # Check that there's still a variable to plot
            sns.regplot(x=columns[k], y="price", data=housing, ax=ax[i][j], color="green")
            ax[i][j].grid(True)
            k += 1
plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()

#%%

fig, axs = plt.subplots(ncols=2, nrows=11, figsize=(20, 50))  # Adjust to create 22 subplots
columns = ['grade', 'waterfront', 'sqft_living', 'Amazon_HQ_distance', 'lat', 'view', 'Starbucks_distance', 'Boeing_Plant_distance', 'sqft_living15', 'Microsoft_distance', 'sqft_product', 'sqft_above', 'year', 'yr_built', 'zipcode', 'bathrooms', 'condition', 'long', 'sqft_lot15', 'yr_renovated', 'floors', 'sqft_lot']
index = 0
axs = axs.flatten()
for k in columns:
    sns.boxplot(y=k, data=housing, ax=axs[index], color="yellow")
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

for k in columns:
    v = housing[k]
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(housing)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))

#%%
# split data into training, validation, and testing sets
X_set, X_test, y_set, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_set, y_set, test_size=0.2)

# create the xgboost matrices
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

evals = [(dtrain, "train"), (dval, "valid")]
npt_callback = NeptuneCallback(run=run)

# create the parameter grid
params_grid = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "reg:squarederror",
    "seed": 42,
}

# train the model
model = xgb.train(
    params=params_grid,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=10,
    callbacks=[npt_callback],
)

# create the xgboost matrix for the test data
dtest = xgb.DMatrix(X_test, label=y_test)

# make predictions on the test data
y_pred = model.predict(dtest)

# calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# calculate the root mean squared error
rmse = sqrt(mse)

# calculate the r2 score
r2 = r2_score(y_test, y_pred)

# log the metrics to neptune
run["mae"].log(mae)
run["mse"].log(mse)
run["rmse"].log(rmse)
run["r2"].log(r2)

# log the model to neptune
run["model"].upload(File.as_pickle(model))

# log the feature importances to neptune
fig, ax = plt.subplots(figsize=(12, 18))
plot_importance(model, ax=ax)
run["feature_importances"].log(neptune.types.File.as_image(fig))

# log the feature correlations to neptune
corr = X.corr()
fig, ax = plt.subplots(figsize=(12, 18))
sns.heatmap(corr, annot=True, ax=ax)
run["feature_correlations"].log(neptune.types.File.as_image(fig))

# log the feature correlations to neptune
fig, ax = plt.subplots(figsize=(12, 18))
plot_tree(model, ax=ax)
run["tree"].log(neptune.types.File.as_image(fig))

# log the feature correlations to neptune
fig, ax = plt.subplots(figsize=(12, 18))
to_graphviz(model, ax=ax)
run["graphviz"].log(neptune.types.File.as_image(fig))

# end the neptune run
run.stop()
