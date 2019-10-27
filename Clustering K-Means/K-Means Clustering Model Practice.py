# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:08:48 2019

@author: mhossa12
"""

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
#dataset = pd.read_csv('4Clustered_Points.csv')
#X = dataset.iloc[:, 0:2].values

#dataset = pd.read_csv('CompSurvey_BonusPoints.csv')
#X = dataset.iloc[:, 0:9].values

dataset = pd.read_excel('BacteriaProject_Training_Data.xlsx')
X = dataset.iloc[:, 1:3].values


"""
Visualize the row dataset into two dimensional scatter splot.

"""
#X_ordinate = dataset.iloc[:, 0].values
#Y_ordinate = dataset.iloc[:, 1].values
#plt.scatter(X_ordinate,Y_ordinate)  
plt.scatter(X[:, 0], X[:, 1])
plt.title("Scatter Plot Visualization with the Raw data")
plt.xlabel("X_ordinate values")
plt.ylabel("Y_ordinate values")
plt.grid()
plt.show()

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=4):
    # 1. Randomly choose clusters
    rnd_num = np.random.RandomState(rseed)
    
    i = rnd_num.permutation(X.shape[0])[:n_clusters]
    print ("i:", i)
    
    centers = X[i]
    print ("centers:", centers)
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        print ("labels.size:", labels.size)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        print ("New Centers:", new_centers)
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
#        print ("Re Centers:", centers)
        
    return centers, labels

centers, labels = find_clusters(X, 6)
print ("Finalized centers:", centers)
#print ("Finalized labels:", labels)
print ("Finalized labels.size:", labels.size)
    
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis');
plt.title("Scatter Plot Visualization of the clustered data")
plt.xlabel("X_ordinate values")
plt.ylabel("Y_ordinate values")
plt.grid()
plt.show()
            