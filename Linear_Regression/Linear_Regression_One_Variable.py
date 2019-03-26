# -*- coding: utf-8 -*-
# Author Kishor Bhosale

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error,r2_score

#Read the data
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

#Remove Nan
train_dataset = train_dataset.dropna()
test_dataset = test_dataset.dropna()

#loading the data in numpy matrix 
x_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, 1].values
x_test = test_dataset.iloc[:, :-1].values
y_test = test_dataset.iloc[:, 1].values


regressor = LinearRegression().fit(x_train,y_train)

y_pred = regressor.predict(x_test)

error = y_test-y_pred

print("Coefficient : ", regressor.coef_)
print("Mean squared error : ", mean_squared_error(y_test,y_pred))
print("r2_Score : ", r2_score(y_test,y_pred))


# Visualize the results
plt.scatter(x_train,y_train,color="red")
plt.title(" train data set ")
plt.xlabel("X_train")
plt.ylabel("Y_train")
plt.show()


plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title(" (Test data set)")
plt.xlabel("X_test")
plt.ylabel("Y_test")
plt.show()