import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# loading in the bikes csv
bikes_df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')
bikes_mini = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/biking_holdout_test_mini.csv')
bikes_holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes_december.csv')

# check out the info
bikes_df.info()

# Preprocessing 
bikes_df.head()
bikes_df['dteday'] = pd.to_datetime(bikes_df['dteday'], format='%m/%d/%y')
bikes_mini['dteday'] = pd.to_datetime(bikes_mini['dteday'], format='%m/%d/%y')
bikes_holdout['dteday'] = pd.to_datetime(bikes_holdout['dteday'], format='%m/%d/%y')

# create new features - year, month, day, dayofweek
bikes_df['year'] = pd.to_datetime(bikes_df['dteday']).dt.year
bikes_df['month'] = pd.to_datetime(bikes_df['dteday']).dt.month
bikes_df['day'] = pd.to_datetime(bikes_df['dteday']).dt.day
bikes_df['dayofweek'] = pd.to_datetime(bikes_df['dteday']).dt.dayofweek

bikes_mini['year'] = pd.to_datetime(bikes_mini['dteday']).dt.year
bikes_mini['month'] = pd.to_datetime(bikes_mini['dteday']).dt.month
bikes_mini['day'] = pd.to_datetime(bikes_mini['dteday']).dt.day
bikes_mini['dayofweek'] = pd.to_datetime(bikes_mini['dteday']).dt.dayofweek

bikes_holdout['year'] = pd.to_datetime(bikes_holdout['dteday']).dt.year
bikes_holdout['month'] = pd.to_datetime(bikes_holdout['dteday']).dt.month
bikes_holdout['day'] = pd.to_datetime(bikes_holdout['dteday']).dt.day
bikes_holdout['dayofweek'] = pd.to_datetime(bikes_holdout['dteday']).dt.dayofweek

# drop dteday column
bikes_df = bikes_df.drop('dteday', axis=1)
bikes_mini = bikes_mini.drop('dteday', axis=1)
bikes_holdout = bikes_holdout.drop('dteday', axis=1)

# one hot encoding
categorical_features = ['season', 'hr', 'holiday', 'workingday', 'weathersit', 'year', 'month', 'day', 'dayofweek']
bikes_df = pd.get_dummies(bikes_df, columns=categorical_features, dtype=int)
bikes_mini = pd.get_dummies(bikes_mini, columns=categorical_features, dtype=int)
bikes_holdout = pd.get_dummies(bikes_holdout, columns=categorical_features, dtype=int)

# min max scaling
scaler = MinMaxScaler()
bikes_df[['temp_c', 'hum', 'feels_like_c', 'windspeed']] = scaler.fit_transform(bikes_df[['temp_c', 'hum', 'feels_like_c', 'windspeed']])
bikes_mini[['temp_c', 'hum', 'feels_like_c', 'windspeed']] = scaler.fit_transform(bikes_mini[['temp_c', 'hum', 'feels_like_c', 'windspeed']])
bikes_holdout[['temp_c', 'hum', 'feels_like_c', 'windspeed']] = scaler.fit_transform(bikes_holdout[['temp_c', 'hum', 'feels_like_c', 'windspeed']])

# creating a total count column
bikes_df['total_count'] = bikes_df['casual'] + bikes_df['registered']
# when it comes to adding these columns together, we don't care about the specifics between casual and registered, we just want the total count, as it provides more information
# additionally, the questions we need to answer surround total count, not casual or registered
# drop casual and registered columns
bikes_df = bikes_df.drop(columns=['casual', 'registered'])
# it's important to scale the numbers as not scaling would cause the model to think that the total count is more important than the other features
# scale bikes_df total_count
# bikes_df[['total_count']] = scaler.fit_transform(bikes_df[['total_count']])
# features and the target
X = bikes_df.drop(columns=['total_count'])
y = bikes_df['total_count']

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape