
#%%

import pandas as pd

customers = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
customers.head()



#%%
#customers['age'].value_counts() #No wierd types
#customers['marital'].value_counts() #unknows = 69
#customers['job'].value_counts() #unknowns = 294
#customers['education'].value_counts() #unknowns =1535
#customers['default'].value_counts() #unknows = 7725
#customers['housing'].value_counts() #unknows = 894
#customers['loan'].value_counts() #unknows = 894
# customers['y'].value_counts() #unknows = 894

#%%
#Cleaning all the unknowns in the first 7 columns
testClean = customers
testClean["job"] = testClean["job"].replace(['unknown'], "admin.")
testClean["marital"] = testClean["marital"].replace(['unknown'], "married")
testClean["education"] = testClean["education"].replace(['unknown'], "university.degree")
testClean["default"] = testClean["default"].replace(['unknown'], "no")
testClean["housing"] = testClean["housing"].replace(['unknown'], "yes")
testClean["loan"] = testClean["loan"].replace(['unknown'], "no")
testClean.head()