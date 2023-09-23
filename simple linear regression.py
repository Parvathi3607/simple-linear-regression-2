# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:29:52 2023

@author: Parva
"""
## simple linear regression

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# np is the shortcut name of numpy
# pd is the shortcut name of pandas
# plt is the shortcut name of matplotlib

# import the dataset

dataset = pd.read_csv(r"C:\Users\Parva\OneDrive\Desktop\naresh it\11th (2)\11th\SIMPLE LINEAR REGRESSION\Salary_Data.csv")
X = dataset.iloc[:,:-1].values
# iloc returns a pandas series when one row is selected(:-1 exclude column from right side)

Y = dataset.iloc[:,1].values
# :colon will read from first to last column dataset(3 to get the only)



# spliting the dataset in training set & testing set

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.20, random_state=0) # test_size=0.30 , test_size=0.25.

# fitting simple linear regression to the traning set & specific library 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# predicting the test result

Y_pred = regressor.predict(X_test)

# visualising the training set results
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train , regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


    
# visualising the test set results
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
