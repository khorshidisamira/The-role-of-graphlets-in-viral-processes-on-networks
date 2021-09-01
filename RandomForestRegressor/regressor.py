# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:03:32 2018

@author: Samira
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf

matplotlib.style.use('ggplot') 
 

#dataset = pd.read_csv('sample_graphs_GFD_ass_hawkes_eigen_sis.csv', nrows = 2454) #All

#dataset = pd.read_csv('retweet_GFD_ass_hawkes_eigen_sis.csv', header=0,nrows = 498) #rt-retweet
#dataset = pd.read_csv('karate_GFD_ass_hawkes_eigen_sis.csv', header=0,nrows = 500) #karate
#dataset = pd.read_csv('dolphine_GFD_ass_hawkes_eigen_sis.csv', header=0,nrows = 500) # dolphins
#dataset = pd.read_csv('hitech_GFD_ass_hawkes_eigen_sis.csv', header=0,nrows = 500) #hi-tech
dataset = pd.read_csv('Caltech_GFD_ass_hawkes_eigen_sis.csv', header=0,nrows = 500) #Caltech

 
x_ass = dataset.iloc[:, 2].values # assortivity 
x_ass_GFD = dataset.iloc[:, 2:32].values # GDF + assortivity  
x_GFD = dataset.iloc[:, 3:32].values # GDF 

y_SIS = dataset.iloc[:, -1].values #SIS 
y_Eigen = dataset.iloc[:, -2].values #Eigen 
y_Hawkes = dataset.iloc[:, -3].values #Hawkes 

####################################FORMULA START############################### 
"""
formula_SIS_ass_GFD = "SIS ~ Assortivity + GFD1 + GFD2 + GFD3 + GFD4 + GFD5 + GFD6 + GFD7 + GFD8 + GFD9 + GFD10 + GFD11 + GFD12 + GFD13 + GFD14 + GFD15 + GFD16 + GFD17 + GFD18 + GFD19 + GFD20 + GFD21 + GFD22 + GFD23 + GFD24 + GFD25 + GFD26 + GFD27 + GFD28 + GFD29"
formula_Eigen_ass_GFD = "Eigen ~ Assortivity + GFD1 + GFD2 + GFD3 + GFD4 + GFD5 + GFD6 + GFD7 + GFD8 + GFD9 + GFD10 + GFD11 + GFD12 + GFD13 + GFD14 + GFD15 + GFD16 + GFD17 + GFD18 + GFD19 + GFD20 + GFD21 + GFD22 + GFD23 + GFD24 + GFD25 + GFD26 + GFD27 + GFD28 + GFD29"
formula_Hawkes_ass_GFD = "Hawkes ~ Assortivity + GFD1 + GFD2 + GFD3 + GFD4 + GFD5 + GFD6 + GFD7 + GFD8 + GFD9 + GFD10 + GFD11 + GFD12 + GFD13 + GFD14 + GFD15 + GFD16 + GFD17 + GFD18 + GFD19 + GFD20 + GFD21 + GFD22 + GFD23 + GFD24 + GFD25 + GFD26 + GFD27 + GFD28 + GFD29"
"""
####################################FORMULA END###############################                     
                       
# Splitting the dataset into the Training set and Test set
"""
#################################Assortivity Start#############################
x_SIS_ass_train, x_SIS_ass_test, y_SIS_ass_train, y_SIS_ass_test = train_test_split(x_ass, y_SIS, test_size = 0.3, random_state = 0)
x_Eigen_ass_train, x_Eigen_ass_test, y_Eigen_ass_train, y_Eigen_ass_test = train_test_split(x_ass, y_Eigen, test_size = 0.3, random_state = 0)
x_Hawkes_ass_train, x_Hawkes_ass_test, y_Hawkes_ass_train, y_Hawkes_ass_test = train_test_split(x_ass, y_Hawkes, test_size = 0.3, random_state = 0)
#################################Assortivity END###############################

########################Assortivity with GFD Start#############################
x_SIS_ass_GFD_train, x_SIS_ass_GFD_test, y_SIS_ass_GFD_train, y_SIS_ass_GFD_test = train_test_split(x_ass_GFD, y_SIS, test_size = 0.3, random_state = 0)
x_Eigen_ass_GFD_train, x_Eigen_ass_GFD_test, y_Eigen_ass_GFD_train, y_Eigen_ass_GFD_test = train_test_split(x_ass_GFD, y_Eigen, test_size = 0.3, random_state = 0)
x_Hawkes_ass_GFD_train, x_Hawkes_ass_GFD_test, y_Hawkes_ass_GFD_train, y_Hawkes_ass_GFD_test = train_test_split(x_ass_GFD, y_Hawkes, test_size = 0.3, random_state = 0)
########################Assortivity with GFD End###############################
"""

########################GFD Start#############################
x_SIS_GFD_train, x_SIS_GFD_test, y_SIS_GFD_train, y_SIS_GFD_test = train_test_split(x_GFD, y_SIS, test_size = 0.3, random_state = 0)
x_Eigen_GFD_train, x_Eigen_GFD_test, y_Eigen_GFD_train, y_Eigen_GFD_test = train_test_split(x_GFD, y_Eigen, test_size = 0.3, random_state = 0)
x_Hawkes_GFD_train, x_Hawkes_GFD_test, y_Hawkes_GFD_train, y_Hawkes_GFD_test = train_test_split(x_GFD, y_Hawkes, test_size = 0.3, random_state = 0)
########################GFD End###############################


# Fitting Linear Regression to the dataset for three target with GFD

regressor_SIS_GFD_linear = LinearRegression()
regressor_SIS_GFD_linear.fit(x_SIS_GFD_train, y_SIS_GFD_train)

# Predicting a new result for SIS based on assortivity
y_SIS_GFD_pred_linear = regressor_SIS_GFD_linear.predict(x_SIS_GFD_test)

#Score and MSE & COEF
coef_SIS_GFD_linear = regressor_SIS_GFD_linear.coef_

score_SIS_GFD_linear = regressor_SIS_GFD_linear.score(x_SIS_GFD_test, y_SIS_GFD_test)
mse_SIS_GFD_linear =  mean_squared_error(y_SIS_GFD_test, y_SIS_GFD_pred_linear) 
#################################SIS END#######################################


#################################Eigen Start###################################
regressor_Eigen_GFD_linear = LinearRegression()
regressor_Eigen_GFD_linear.fit(x_Eigen_GFD_train, y_Eigen_GFD_train)

# Predicting a new result for SIS based on assortivity
y_Eigen_GFD_pred_linear = regressor_Eigen_GFD_linear.predict(x_Eigen_GFD_test)

#Score and MSE & COEF
coef_Eigen_GFD_linear = regressor_Eigen_GFD_linear.coef_
 
score_Eigen_GFD_linear = regressor_Eigen_GFD_linear.score(x_Eigen_GFD_test, y_Eigen_GFD_test)

mse_Eigen_GFD_linear =  mean_squared_error(y_Eigen_GFD_test, y_Eigen_GFD_pred_linear) 
#################################Eigen END#####################################


#################################Hawkes Start##################################
regressor_Hawkes_GFD_linear = LinearRegression()
regressor_Hawkes_GFD_linear.fit(x_Hawkes_GFD_train, y_Hawkes_GFD_train)

# Predicting a new result for Hawkes based on assortivity
y_Hawkes_GFD_pred_linear = regressor_Hawkes_GFD_linear.predict(x_Hawkes_GFD_test)

#Score and MSE & COEF
coef_Hawkes_GFD_linear = regressor_Hawkes_GFD_linear.coef_

score_Hawkes_GFD_linear = regressor_Hawkes_GFD_linear.score(x_Hawkes_GFD_test, y_Hawkes_GFD_test)

mse_Hawkes_GFD_linear =  mean_squared_error(y_Hawkes_GFD_test, y_Hawkes_GFD_pred_linear)
#################################Hawkes END####################################

# Fitting Linear Regression to the dataset for three target with Assortivity
###############################Assortivity#####################################
#################################SIS Start##################################### 
"""
x_SIS_ass_train_size = x_SIS_ass_train.shape[0]
x_SIS_ass_test_size = x_SIS_ass_test.shape[0]
x_SIS_ass_train = x_SIS_ass_train.reshape((x_SIS_ass_train_size,1))
x_SIS_ass_test = x_SIS_ass_test.reshape((x_SIS_ass_test_size,1))

regressor_SIS_ass = LinearRegression()
regressor_SIS_ass.fit(x_SIS_ass_train, y_SIS_ass_train)

# Predicting a new result for SIS based on assortivity
y_SIS_ass_pred = regressor_SIS_ass.predict(x_SIS_ass_test)

#Score and MSE & COEF
coef_SIS_ass = regressor_SIS_ass.coef_

score_SIS_ass = regressor_SIS_ass.score(x_SIS_ass_test, y_SIS_ass_test)

mse_SIS_ass =  mean_squared_error(y_SIS_ass_test, y_SIS_ass_pred) 
"""

#################################SIS END#######################################

#################################Eigen Start###################################
"""
x_Eigen_ass_train_size = x_Eigen_ass_train.shape[0]
x_Eigen_ass_test_size = x_Eigen_ass_test.shape[0]
x_Eigen_ass_train = x_Eigen_ass_train.reshape((x_Eigen_ass_train_size,1))
x_Eigen_ass_test = x_Eigen_ass_test.reshape((x_Eigen_ass_test_size,1))

regressor_Eigen_ass = LinearRegression()
regressor_Eigen_ass.fit(x_Eigen_ass_train, y_Eigen_ass_train)

# Predicting a new result for Eigen values based on assortivity
y_Eigen_ass_pred = regressor_Eigen_ass.predict(x_Eigen_ass_test)

#Score and MSE & COEF
coef_Eigen_ass = regressor_Eigen_ass.coef_

score_Eigen_ass = regressor_Eigen_ass.score(x_Eigen_ass_test, y_Eigen_ass_test)

mse_Eigen_ass =  mean_squared_error(y_Eigen_ass_test, y_Eigen_ass_pred) 
"""
#################################Eigen END#####################################

#################################Hawkes Start##################################
"""
x_Hawkes_ass_train_size = x_Hawkes_ass_train.shape[0]
x_Hawkes_ass_test_size = x_Hawkes_ass_test.shape[0]
x_Hawkes_ass_train = x_Hawkes_ass_train.reshape((x_Hawkes_ass_train_size,1))
x_Hawkes_ass_test = x_Hawkes_ass_test.reshape((x_Hawkes_ass_test_size,1))

regressor_Hawkes_ass = LinearRegression()
regressor_Hawkes_ass.fit(x_Hawkes_ass_train, y_Hawkes_ass_train)

# Predicting a new result for Hawkes based on assortivity
y_Hawkes_ass_pred = regressor_Hawkes_ass.predict(x_Hawkes_ass_test)

#Score and MSE & COEF
coef_Hawkes_ass = regressor_Hawkes_ass.coef_

score_Hawkes_ass = regressor_Hawkes_ass.score(x_Hawkes_ass_test, y_Hawkes_ass_test)

mse_Hawkes_ass =  mean_squared_error(y_Hawkes_ass_test, y_Hawkes_ass_pred) 

#################################Hawkes END####################################


# Fitting Random Forest Regression to the dataset for three target with Assortivity
###############################Assortivity#####################################
#################################SIS Start##################################### 
 
regressor_SIS_ass_RF = RandomForestRegressor(n_estimators = 2000, random_state = 0) #n_estimators: number of trees in the forest
regressor_SIS_ass_RF.fit(x_SIS_ass_train, y_SIS_ass_train)

# Predicting a new result for SIS based on assortivity
y_SIS_ass_pred_RF = regressor_SIS_ass_RF.predict(x_SIS_ass_test)

#Score and MSE 
score_SIS_ass_RF = regressor_SIS_ass_RF.score(x_SIS_ass_test, y_SIS_ass_test)

mse_SIS_ass_RF =  mean_squared_error(y_SIS_ass_test, y_SIS_ass_pred_RF) 

#################################SIS END#######################################


#################################Eigen Start###################################
 
regressor_Eigen_ass_RF = RandomForestRegressor(n_estimators = 2000, random_state = 0) 
regressor_Eigen_ass_RF.fit(x_Eigen_ass_train, y_Eigen_ass_train)

# Predicting a new result for Eigen values based on assortivity
y_Eigen_ass_pred_RF = regressor_Eigen_ass_RF.predict(x_Eigen_ass_test)

#Score and MSE 
score_Eigen_ass_RF = regressor_Eigen_ass_RF.score(x_Eigen_ass_test, y_Eigen_ass_test)

mse_Eigen_ass_RF =  mean_squared_error(y_Eigen_ass_test, y_Eigen_ass_pred_RF) 

#################################Eigen END#####################################

#################################Hawkes Start################################## 

regressor_Hawkes_ass_RF = RandomForestRegressor(n_estimators = 2000, random_state = 0) 
regressor_Hawkes_ass_RF.fit(x_Hawkes_ass_train, y_Hawkes_ass_train)

# Predicting a new result for Hawkes based on assortivity
y_Hawkes_ass_pred_RF = regressor_Hawkes_ass_RF.predict(x_Hawkes_ass_test)

#Score and MSE 
score_Hawkes_ass_RF = regressor_Hawkes_ass_RF.score(x_Hawkes_ass_test, y_Hawkes_ass_test)

mse_Hawkes_ass_RF =  mean_squared_error(y_Hawkes_ass_test, y_Hawkes_ass_pred_RF) 

#################################Hawkes END####################################  


# Fitting Random Forest Regression to the dataset for three target with Assortivity and GFD
###############################Assortivity#####################################
#################################SIS Start#####################################

regressor_SIS_ass_GFD = RandomForestRegressor(n_estimators = 2000, random_state = 0,max_features=9, criterion = "mse") #n_estimators: number of trees in the forest
regressor_SIS_ass_GFD.fit(x_SIS_ass_GFD_train, y_SIS_ass_GFD_train)

# Predicting a new result for SIS based on assortivity
y_SIS_ass_GFD_pred = regressor_SIS_ass_GFD.predict(x_SIS_ass_GFD_test)

#Score and MSE 
score_SIS_ass_GFD = regressor_SIS_ass_GFD.score(x_SIS_ass_GFD_test, y_SIS_ass_GFD_test)

mse_SIS_ass_GFD =  mean_squared_error(y_SIS_ass_GFD_test, y_SIS_ass_GFD_pred) 

#################################SIS END#######################################

#################################Eigen Start###################################

regressor_Eigen_ass_GFD = RandomForestRegressor(n_estimators = 2000, random_state = 0,max_features=9, criterion = "mse") #n_estimators: number of trees in the forest
regressor_Eigen_ass_GFD.fit(x_Eigen_ass_GFD_train, y_Eigen_ass_GFD_train)

# Predicting a new result for SIS based on assortivity
y_Eigen_ass_GFD_pred = regressor_Eigen_ass_GFD.predict(x_Eigen_ass_GFD_test)

#Score and MSE 
score_Eigen_ass_GFD = regressor_Eigen_ass_GFD.score(x_Eigen_ass_GFD_test, y_Eigen_ass_GFD_test)

mse_Eigen_ass_GFD =  mean_squared_error(y_Eigen_ass_GFD_test, y_Eigen_ass_GFD_pred) 
 
#################################Eigen END#####################################

#################################Hawkes Start##################################
regressor_Hawkes_ass_GFD = RandomForestRegressor(n_estimators = 2000, random_state = 0,max_features=9, criterion = "mse") #n_estimators: number of trees in the forest
regressor_Hawkes_ass_GFD.fit(x_Hawkes_ass_GFD_train, y_Hawkes_ass_GFD_train)

# Predicting a new result for SIS based on assortivity
y_Hawkes_ass_GFD_pred = regressor_Hawkes_ass_GFD.predict(x_Hawkes_ass_GFD_test)

#Score and MSE 
score_Hawkes_ass_GFD = regressor_Hawkes_ass_GFD.score(x_Hawkes_ass_GFD_test, y_Hawkes_ass_GFD_test)

mse_Hawkes_ass_GFD =  mean_squared_error(y_Hawkes_ass_GFD_test, y_Hawkes_ass_GFD_pred)  
#################################Hawkes END####################################
"""


# Fitting Linear Regression to the dataset for three target with Assortivity and GFD
###############################Assortivity#####################################
#################################SIS Start#####################################
"""
regressor_SIS_ass_GFD_linear = LinearRegression()
regressor_SIS_ass_GFD_linear.fit(x_SIS_ass_GFD_train, y_SIS_ass_GFD_train)

# Predicting a new result for SIS based on assortivity
y_SIS_ass_GFD_pred_linear = regressor_SIS_ass_GFD_linear.predict(x_SIS_ass_GFD_test)

#Score and MSE & COEF
coef_SIS_ass_GFD_linear = regressor_SIS_ass_GFD_linear.coef_

score_SIS_ass_GFD_linear = regressor_SIS_ass_GFD_linear.score(x_SIS_ass_GFD_test, y_SIS_ass_GFD_test)
mse_SIS_ass_GFD_linear =  mean_squared_error(y_SIS_ass_GFD_test, y_SIS_ass_GFD_pred_linear) 
"""
#################################SIS END#######################################


#################################Eigen Start###################################
"""
regressor_Eigen_ass_GFD_linear = LinearRegression()
regressor_Eigen_ass_GFD_linear.fit(x_Eigen_ass_GFD_train, y_Eigen_ass_GFD_train)

# Predicting a new result for SIS based on assortivity
y_Eigen_ass_GFD_pred_linear = regressor_Eigen_ass_GFD_linear.predict(x_Eigen_ass_GFD_test)

#Score and MSE & COEF
coef_Eigen_ass_GFD_linear = regressor_Eigen_ass_GFD_linear.coef_
 
score_Eigen_ass_GFD_linear = regressor_Eigen_ass_GFD_linear.score(x_Eigen_ass_GFD_test, y_Eigen_ass_GFD_test)

mse_Eigen_ass_GFD_linear =  mean_squared_error(y_Eigen_ass_GFD_test, y_Eigen_ass_GFD_pred_linear) 
"""
#################################Eigen END#####################################


#################################Hawkes Start##################################
"""
regressor_Hawkes_ass_GFD_linear = LinearRegression()
regressor_Hawkes_ass_GFD_linear.fit(x_Hawkes_ass_GFD_train, y_Hawkes_ass_GFD_train)

# Predicting a new result for Hawkes based on assortivity
y_Hawkes_ass_GFD_pred_linear = regressor_Hawkes_ass_GFD_linear.predict(x_Hawkes_ass_GFD_test)

#Score and MSE & COEF
coef_Hawkes_ass_GFD_linear = regressor_Hawkes_ass_GFD_linear.coef_

score_Hawkes_ass_GFD_linear = regressor_Hawkes_ass_GFD_linear.score(x_Hawkes_ass_GFD_test, y_Hawkes_ass_GFD_test)

mse_Hawkes_ass_GFD_linear =  mean_squared_error(y_Hawkes_ass_GFD_test, y_Hawkes_ass_GFD_pred_linear) 
#################################Hawkes END####################################



lm_SIS_ass_GFD = smf.ols(formula=formula_SIS_ass_GFD, data=dataset).fit()
 
# print the coefficients
print("###############################SIS regressor coefficients:###############################")
print(lm_SIS_ass_GFD.params)
#print("###############################SIS regressor Summary:###############################")
#print(lm_SIS_ass_GFD.summary())

# only include small P values in the model
# instantiate and fit model
#significant_SIS_formula = "SIS ~ Assortivity + GFD6 + GFD7 + GFD8 + GFD9 + GFD10 + GFD11 + GFD13 + GFD14 + GFD15 + GFD19 + GFD22 + GFD23 + GFD24 + GFD27 + GFD29" # All networks
#significant_SIS_formula = "SIS ~ Assortivity + GFD10 + GFD18 + GFD22 + GFD24 + GFD25 + GFD26 + GFD27 + GFD28 + GFD29" # rt-retweet
#significant_SIS_formula = "SIS ~ Assortivity + GFD1 + GFD3 + GFD4 + GFD9 + GFD10 + GFD11 + GFD12 + GFD13 + GFD14 + GFD16 + GFD18 + GFD19 + GFD21 + GFD28 + GFD29" # karate
#significant_SIS_formula = "SIS ~ Assortivity + GFD10 + GFD12 + GFD13 + GFD28" # dolphines
#significant_SIS_formula = "SIS ~ Assortivity + GFD1 + GFD3 + GFD4 + GFD6 + GFD9 + GFD10 + GFD11 + GFD12 + GFD13 + GFD14 + GFD16 + GFD17 + GFD18 + GFD19" # hi-tewch
significant_SIS_formula = "SIS ~ Assortivity" # Caltech

significant_lm_SIS_ass_GFD = smf.ols(formula=significant_SIS_formula, data=dataset).fit()
# calculate r-square 
significant_SIS_rsquared = significant_lm_SIS_ass_GFD.rsquared





lm_Eigen_ass_GFD = smf.ols(formula=formula_Eigen_ass_GFD, data=dataset).fit()
 
# print the coefficients
print("###############################Eigen regressor coefficients:###############################")
print(lm_Eigen_ass_GFD.params)
#print("###############################Eigen regressor Summary:")
#print(lm_Eigen_ass_GFD.summary())

# only include small P values in the model
# instantiate and fit model
#significant_Eigen_formula = "Eigen ~ GFD1 + GFD3 + GFD4 + GFD5 + GFD6 + GFD8 + GFD9 + GFD10 + GFD11 + GFD13 + GFD15 + GFD16 + GFD19 + GFD21 + GFD22 + GFD23 + GFD26 + GFD27" # All networks
#significant_Eigen_formula = "Eigen ~ Assortivity + GFD10 + GFD14 + GFD15 + GFD20 + GFD21 + GFD22 + GFD24 + GFD25 + GFD28 + GFD29" # rt-retweet
#significant_Eigen_formula = "Eigen ~ Assortivity + GFD3 + GFD10 + GFD12 + GFD14 + GFD15 + GFD16 + GFD18 + GFD19 + GFD24 + GFD27 + GFD28 + GFD29" # karate
#significant_Eigen_formula = "Eigen ~ Assortivity + GFD10 + GFD21 + GFD27 + GFD29" # dolphines
#significant_Eigen_formula = "Eigen ~ Assortivity + GFD3 + GFD10 + GFD12 + GFD15 + GFD16 + GFD19 + GFD25 + GFD27+ GFD29" # hi-tech
significant_Eigen_formula = "Eigen ~ Assortivity" # Caltech

significant_lm_Eigen_ass_GFD = smf.ols(formula=significant_Eigen_formula, data=dataset).fit()
# calculate r-square 
significant_Eigen_rsquared = significant_lm_Eigen_ass_GFD.rsquared



lm_Hawkes_ass_GFD = smf.ols(formula=formula_Hawkes_ass_GFD, data=dataset).fit()
 
# print the coefficients
print("###############################Hawkes regressor coefficients:###############################")
print(lm_Hawkes_ass_GFD.params)
#print("###############################Hawkes regressor Summary:###############################")
#print(lm_Hawkes_ass_GFD.summary())

# only include small P values in the model
# instantiate and fit model
#significant_Hawkes_formula = "Hawkes ~ Assortivity + GFD11 + GFD29" # All networks
#significant_Hawkes_formula = "Hawkes ~ GFD26" # rt-retweet
#significant_Hawkes_formula = "Hawkes ~ GFD9 + GFD10 + GFD11 + GFD12 + GFD14" # karate
#significant_Hawkes_formula = "Hawkes ~ Assortivity + GFD10 + GFD12 + GFD13 + GFD14 + GFD28" # dolphines
#significant_Hawkes_formula = "Hawkes ~ Assortivity + GFD1 + GFD3 + GFD4 + GFD6 + GFD9 + GFD10 + GFD11 + GFD12 + GFD13 + GFD14 + GFD16 + GFD17 + GFD18 + GFD19" # hi-tech
significant_Hawkes_formula = "Hawkes ~ Assortivity" # Caltech

significant_lm_Hawkes_ass_GFD = smf.ols(formula=significant_Hawkes_formula, data=dataset).fit()
# calculate r-square 
significant_Hawkes_rsquared = significant_lm_Hawkes_ass_GFD.rsquared
"""