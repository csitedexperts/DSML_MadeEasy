# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:45:41 2019

@author: mhossa12
"""

# Developing a Logistic Regression Model for a MIV Datasets using LinearRegression Library Modules in Python

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('CompSurvey_Product1.csv')
X = dataset.iloc[:, [1, 7]].values
y = dataset.iloc[:, 8].values


# Splitting the dataset into the Training set and Test set
from sklearn import cross_validation as cv

# Splitting the data sets into Training and Test sets
X_Train, X_Test = cv.train_test_split(X, test_size = .2, random_state = 0)
y_Train, y_Test = cv.train_test_split(y, train_size = .8, random_state = 0)
# Training  => 80% 
## Just keelping backup copies
X_Train0, X_Test0 = X_Train, X_Test
y_Train0, y_Test0 = y_Train, y_Test 

# Implementing the Feature Scaling for Data Normalization
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as pp
sc = pp.StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_Train, y_Train)

# General Classifier codes here 


# Predicting the Test set results
y_pred = classifier.predict(X_Test)

print (y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confMatrix = confusion_matrix(y_Test, y_pred)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_Train, y_Train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression Model with the Training dataset)')
plt.xlabel('Experience in Year')
plt.ylabel('Estimated Salary')
plt.legend()
plt.grid()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_Test, y_Test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression Model Verification with the Test dataset')
plt.xlabel('Experience in Year')
plt.ylabel('Anual Salary')
plt.legend()
plt.grid()
plt.show()