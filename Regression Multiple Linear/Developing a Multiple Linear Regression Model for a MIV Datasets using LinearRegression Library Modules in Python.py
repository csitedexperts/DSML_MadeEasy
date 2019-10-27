# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:09:03 2019

@author: mhossa12
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
datasetDS = pd.read_csv('CompSalary_Data.csv')
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 4].values

X = datasetDS.iloc[:, 0:7].values ## NOT  X = datasetDS.iloc[:, 7].values
## Since X is a Matrix here
#Or X = dataset.iloc[:, :-1].values
#y = datasetDS.iloc[:, 4:5].values  ## NOT  y = datasetDS.iloc[:, 7].values
## Since y is a Transposed Matrix of the y itself
y = datasetDS.iloc[:, 7].values


# Splitting the dataset into the Training set and Test set
from sklearn import cross_validation as cv

# Splitting the data sets into Training and Test sets
X_train, X_test = cv.train_test_split(X, test_size = .20, random_state = 0)
y_train, y_test = cv.train_test_split(y, train_size = .80, random_state = 0)
# Training  => 80% 

# Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print (y_pred)
# Now Build the optimal model using Backward Elimination Model
import statsmodels.formula.api as sm
X2 = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# X2[0=Constant, 1=Experience, 2=Education, 3=Performace, 4=Dedication, 5=Languages, 6=Gender, 7=Eithnicity]
X_all = X2[:, [0, 1, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_all).fit()
regressor_OLS.summary()

# X2[0=Constant, 1=Experience, 2=Education, 3=Performace, 5=Languages, 6=Gender, 7=Eithnicity]
X_opt1 = X2[:, [0, 1, 2, 3, 5, 6, 7]]  # Removed 4
regressor_OLS = sm.OLS(endog = y, exog = X_opt1).fit()
regressor_OLS.summary()

# X2[0=Constant, 1=Experience, 2=Education, 3=Performace, 5=Languages, 7=Eithnicity]
X_opt2 = X2[:, [0, 1, 2, 3, 5, 7]]  # Removed 4, 6
regressor_OLS = sm.OLS(endog = y, exog = X_opt2).fit()
regressor_OLS.summary()


# X2[0=Constant, 1=Experience, 2=Education, 3=Performace, 5=Languages]
X_opt2 = X2[:, [0, 1, 2, 3, 5]]  # Removed 4, 6, 7
regressor_OLS = sm.OLS(endog = y, exog = X_opt2).fit()
regressor_OLS.summary()


# X2[0=Constant, 1=Experience, 2=Education, 3=Performace]
X_opt2 = X2[:, [0, 1, 2, 3]]  # Removed 4, 6, 7, 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt2).fit()
regressor_OLS.summary()


# X2[0=Constant, 1=Experience, 3=Performace]
X_opt2 = X2[:, [0, 1, 3]]  # Removed 4, 6, 7, 5, 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt2).fit()
regressor_OLS.summary()

