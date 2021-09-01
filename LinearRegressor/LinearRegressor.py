# Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

matplotlib.style.use('ggplot')

"""
# Importing the dataset
dataset = pd.read_csv('samplesWithAssortativity.csv', header=None,nrows = 103)

#degree Distributions:
    
X = dataset.iloc[:, 2:11].values
y = dataset.iloc[:, -1].values
               
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)




# Fitting Linar Regression to the dataset
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

# Predicting a new result
y_pred = regressor.predict(X_test)
"""
########################################################
# Importing the dataset
dataset_GFD = pd.read_csv('samplesWithAssortativity.csv', header=0,nrows = 2458)

X_GFD = dataset_GFD.iloc[:, 1:40].values #dataset_GFD.iloc[:, 2:39].values
y_GFD = dataset_GFD.iloc[:, -1].values
 
# Splitting the dataset into the Training set and Test set
X_GFD_train, X_GFD_test, y_GFD_train, y_GFD_test = train_test_split(X_GFD, y_GFD, test_size = 0.3, random_state = 0)

# Fitting Linar Regression to the dataset
regressor_GFD = LinearRegression()
regressor_GFD.fit(X_GFD_train, y_GFD_train)

# Predicting a new result
y_GFD_pred = regressor_GFD.predict(X_GFD_test)

from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(regressor_GFD, prefit=True)
X_GFD_new = model.transform(X_GFD)

from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(regressor_GFD, prefit=True, threshold="mean")
X_GFD_new = model.transform(X_GFD)


# Fitting Random Forest Regression to the new dataset
new_X_GFD_train, new_X_GFD_test, new_y_GFD_train, new_y_GFD_test = train_test_split(X_GFD_new, y_GFD, test_size = 0.3, random_state = 0)
new_regressor_GFD = LinearRegression()
new_regressor_GFD.fit(new_X_GFD_train, new_y_GFD_train)

new_y_GFD_pred = new_regressor_GFD.predict(new_X_GFD_test)

new_mse =  mean_squared_error(new_y_GFD_test, new_y_GFD_pred) 
new_score = new_regressor_GFD.score(new_X_GFD_train,new_y_GFD_train)
#Returns the coefficient of determination R^2 of the prediction.
#The coefficient R^2 is defined as (1 - u/v), where u is
# the residual sum of squares ((y_true - y_pred) ** 2).sum() and
# v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
#The best possible score is 1.0 and it can be negative (because the model 
#can be arbitrarily worse). A constant model that always predicts the expected value 
#of y, disregarding the input features, would get a R^2 score of 0.0.
GFD_score = regressor_GFD.score( X_GFD_train,y_GFD_train)
#DFD_score = regressor.score(X_train, y_train)
#DFD_mse =  mean_squared_error(y_test, y_pred)  
GFD_mse =  mean_squared_error(y_GFD_test, y_GFD_pred)   

# Visualising the Random Forest Regression results
#df3 = pd.DataFrame(dataset_GFD.iloc[:, 0:41])
#df3[41] = regressor_GFD.predict(X_GFD)
#df3[42] = regressor.predict(X)
#ax = df3.plot.scatter(x=0, y=40, color='Red', label='Original')
#df3.plot(x=0, y=41, color='DarkGreen', label='Regressor with GFD', ax=ax)
#df3.plot(x=0, y=42, color='DarkBlue', label='Regressor with DFD', ax=ax)
