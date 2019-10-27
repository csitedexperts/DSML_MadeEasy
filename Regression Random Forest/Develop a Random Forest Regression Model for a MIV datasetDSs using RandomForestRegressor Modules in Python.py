# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:36:25 2019

@author: mhossa12
"""

# Develop a Random Forest Regression Model for a MIV datasetDSs using LinearRegression Library Modules in Python

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasetDS
datasetDS = pd.read_csv('PosSalary_Data.csv')
X = datasetDS.iloc[:, 1:2].values
y = datasetDS.iloc[:, 2].values


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 12, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.legend()
plt.grid()
plt.show()