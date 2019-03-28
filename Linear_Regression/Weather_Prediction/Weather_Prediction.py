# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 07:27:05 2019

@author: kishor.bhosale
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Weather.csv')

dataset.plot(x='MinTemp', y='MaxTemp', style='o')  
plt.title('MinTemp vs MaxTemp')  
plt.xlabel('MinTemp')  
plt.ylabel('MaxTemp')  
plt.show()

X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression().fit(X_train,y_train)

print(regressor.intercept_)
print(regressor.coef_)


y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten()})

df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red')
plt.show()

print("Mean Absolute Error : ",mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))