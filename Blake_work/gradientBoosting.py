url = 'https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv'

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

dat = pd.read_csv(url)

dat['y'] = np.where(dat['y'] == "yes", 1, 0)

X = dat[['age','nr.employed', 'euribor3m', 'campaign', 'poutcome']]
y = dat['y']


X = pd.get_dummies(X)

print(len(X.index))
print(len(y.index))

# classifier = GradientBoostingClassifier(max_depth = 12)

# x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# classifier.fit(x_train, y_train)

# classifier.predict()

