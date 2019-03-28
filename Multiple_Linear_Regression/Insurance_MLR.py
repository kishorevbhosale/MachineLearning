#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:36:10 2019

@author: kishor.bhosale
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('insurance.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,6].values
 

#Categorical data
lableencoder_x = LabelEncoder()
x[:,1] = lableencoder_x.fit_transform(x[:,1])

lableencoder_x = LabelEncoder()
x[:,4] = lableencoder_x.fit_transform(x[:,4])

lableencoder_x = LabelEncoder()
x[:,5] = lableencoder_x.fit_transform(x[:,5])

onehotencoder = OneHotEncoder(categorical_features=[5])

x = onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap
x = x[:,1:]

# split the data in training and test dataseet 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 0)

# Feature Scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# Applying Multiple Linear regression model to training dataset
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(x_test)


#creating a column in the beginning with all ones i.e x0 values
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((1338,1)).astype(int),values = x, axis=1)


#Applying backward elimination
x_opt = x[:,[0,1,2,3,4,5,6,7,8]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

# for x5 P>\t\   =  0.693 remove x5
x_opt = x[:,[0,1,2,3,4,6,7,8]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

# for x1 P>\t\   =  0.460 remove x1
x_opt = x[:,[0,2,3,4,6,7,8]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


#Applying the model again
x_train,x_test,y_train,y_test = train_test_split(x_opt,y,test_size = 0.20, random_state = 0)
regressor1 = LinearRegression()
regressor1.fit(x_train,y_train)

y_pred2 = regressor1.predict(x_test)