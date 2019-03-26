# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#split the daataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#Fitting simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predict Test result
y_pred = regressor.predict(x_test)

#Plot the training set result
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Training dataset)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#Plot the test set result
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Test dataset)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()