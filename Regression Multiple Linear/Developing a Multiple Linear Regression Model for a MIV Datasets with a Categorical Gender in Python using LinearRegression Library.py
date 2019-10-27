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
datasetDS = pd.read_csv('CompSalary_GenderCategorical.csv')
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 4].values

X = datasetDS.iloc[:, 0:7].values ## NOT  X = datasetDS.iloc[:, 7].values
## Since X is a Matrix here
#Or X = dataset.iloc[:, :-1].values
#y = datasetDS.iloc[:, 4:5].values  ## NOT  y = datasetDS.iloc[:, 7].values
## Since y is a Transposed Matrix of the y itself
y = datasetDS.iloc[:, 7].values

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing as pp
#import LabelEncoder, OneHotEncoder
labelencoder = pp.LabelEncoder()
X[:, 5] = labelencoder.fit_transform(X[:, 5])
onehotencoder = pp.OneHotEncoder(categorical_features = [5])
X1 = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X2 = X1[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn import cross_validation as cv

# Splitting the data sets into Training and Test sets
X_train, X_test = cv.train_test_split(X2, test_size = .20, random_state = 0)
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
X3 = np.append(arr = np.ones((50, 1)).astype(int), values = X2, axis = 1)

# X3[0=Constant, 1=Gender, 2=Experience, 3=Education, 4=Performace, 5=Dedication, 6=Languages, 7=Eithnicity]
X_all = X3[:, [0, 1, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_all).fit()
regressor_OLS.summary()

# X3[0=Constant, 1=Gender, 2=Experience, 3=Education, 4=Performace, 6=Languages, 7=Eithnicity]
X_opt1 = X3[:, [0, 1, 2, 3, 4, 6, 7]]  # Removed 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt1).fit()
regressor_OLS.summary()

# X3[0=Constant, 2=Experience, 3=Education, 4=Performace, 6=Languages, 7=Eithnicity]
X_opt2 = X3[:, [0, 2, 3, 4, 6, 7]]  # Removed 5, 1
regressor_OLS = sm.OLS(endog = y, exog = X_opt2).fit()
regressor_OLS.summary()


# X3[0=Constant, 2=Experience, 3=Education, 4=Performace, 7=Eithnicity]
X_opt3 = X3[:, [0, 2, 3, 4, 7]]  # Removed 5, 1, 6
regressor_OLS = sm.OLS(endog = y, exog = X_opt3).fit()
regressor_OLS.summary()


# X3[0=Constant, 2=Experience, 4=Performace, 7=Eithnicity]
X_opt4 = X3[:, [0, 2, 4, 7]]  # Removed 5, 1, 6, 3
regressor_OLS = sm.OLS(endog = y, exog = X_opt4).fit()
regressor_OLS.summary()


# X3[0=Constant, 2=Experience, 4=Performace]
X_opt5 = X3[:, [0, 2, 4]]  # Removed 5, 1, 6, 3, 7
regressor_OLS = sm.OLS(endog = y, exog = X_opt5).fit()
regressor_OLS.summary()

