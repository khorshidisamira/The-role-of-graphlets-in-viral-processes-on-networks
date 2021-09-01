# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold

matplotlib.style.use('ggplot')

"""
# Importing the dataset
#dataset = pd.read_csv('sample_graphs.csv', header=0,nrows = 500)
#dataset = pd.read_csv('FacebookNetworkAndSamples.csv', header=0,nrows = 603)
dataset = pd.read_csv('samplesWithAssortativity.csv', header=0,nrows = 500)
#dataset = pd.read_csv('samplesWithAssortativityAndEigenValue.csv', header=0,nrows = 500)

#degree Distributions:
    
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, -1].values
 
                        
# Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'nan', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 1:10])
#X[:, 1:10] = imputer.transform(X[:, 1:10])

                
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting Random Forest Regression to the dataset

regressor = RandomForestRegressor(n_estimators = 2000, random_state = 0,max_features=9, criterion = "mse") #n_estimators: number of trees in the forest
regressor.fit(X_train, y_train)


# Predicting a new result
y_pred = regressor.predict(X_test)

"""
########################################################
# Importing the dataset
#dataset_GFD = pd.read_csv('sample_graphs.csv', header=0,nrows = 500)
#dataset_GFD = pd.read_csv('FacebookNetworkAndSamples.csv', header=0,nrows = 603)
#dataset_GFD = pd.read_csv('samplesWithAssortativity.csv', header=0,nrows = 2458)

dataset_GFD = pd.read_csv('samplesWithAssortativityAndEigenValue.csv', header=0,nrows = 500)

x_GFD = dataset_GFD.iloc[:, 2:40].values # without assortivity
#x_GFD = dataset_GFD.iloc[:, 1:40].values # with assortivity
y_GFD = dataset_GFD.iloc[:, -1].values
               
# Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

# Splitting the dataset into the Training set and Test set

X_GFD_train, X_GFD_test, y_GFD_train, y_GFD_test = train_test_split(x_GFD, y_GFD, test_size = 0.3, random_state = 0)

# Fitting Random Forest Regression to the dataset

regressor_GFD = RandomForestRegressor(n_estimators = 2000, random_state = 0,max_features=9, criterion = "mse") #n_estimators: number of trees in the forest
regressor_GFD.fit(X_GFD_train, y_GFD_train)

# Predicting a new result
y_GFD_pred = regressor_GFD.predict(X_GFD_test)


#removing less important features using select from model
model = SelectFromModel(regressor_GFD, prefit=True, threshold="mean")
X_GFD_new = model.transform(x_GFD)


# Fitting Random Forest Regression to the new dataset
new_X_GFD_train, new_X_GFD_test, new_y_GFD_train, new_y_GFD_test = train_test_split(X_GFD_new, y_GFD, test_size = 0.3, random_state = 0)
new_regressor_GFD = RandomForestRegressor(n_estimators = 2000, random_state = 0, criterion = "mse") #n_estimators: number of trees in the forest
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
"""
DFD_score = regressor.score(X_train, y_train)
DFD_mse =  mean_squared_error(y_test, y_pred)
"""  
GFD_mse =  mean_squared_error(y_GFD_test, y_GFD_pred) 
 
# Visualising the Random Forest Regression results
#df3 = pd.DataFrame(dataset_GFD.iloc[:, 0:41])
#df3[41] = regressor_GFD.predict(X_GFD)
#df3[42] = regressor.predict(X)
#ax = df3.plot.scatter(x=0, y=40, color='Red', label='Original')
#df3.plot(x=0, y=41, color='DarkGreen', label='Regressor with GFD', ax=ax)
#df3.plot(x=0, y=42, color='DarkBlue', label='Regressor with DFD', ax=ax)


#kf = KFold(n_splits=10)
subset_size = x_GFD.shape[0]/10
mse_array = []
for i in range(10):
    index = i*subset_size
    X_testing_this_round = x_GFD[index:index + subset_size,:]  
    y_testing_this_round = y_GFD[index:index + subset_size]  
    X_training_this_round = np.concatenate((x_GFD[:index, :], x_GFD[(i+1)*subset_size:, :]), axis=0)
    y_training_this_round = np.concatenate((y_GFD[:index], y_GFD[(i+1)*subset_size:]), axis=0)
    
    fold_regressor = RandomForestRegressor(n_estimators = 2000, random_state = 0, max_features=9, criterion = "mse") #n_estimators: number of trees in the forest
    fold_regressor.fit(X_training_this_round, y_training_this_round)
    fold_y_pred = fold_regressor.predict(X_testing_this_round)
    fold_mse = mean_squared_error(y_testing_this_round, fold_y_pred)
    mse_array.append(fold_mse)
average_mse = sum(mse_array)/len(mse_array)
#extraction of feature importance
importances = regressor_GFD.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor_GFD.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure(figsize=(12,16))
plt.title("Feature importances")
plt.bar(range(x_GFD.shape[1]), importances[indices], color="r", yerr=std[indices])
plt.xticks(range(x_GFD.shape[1]), indices)
plt.xlim([-1, x_GFD.shape[1]])
plt.xlabel("Feature indices")
plt.ylabel("Feature importance")
plt.show()
                        