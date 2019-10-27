# Hierarchical Clustering
# Develop and Verify a Hierarchical Clustering Model in Python

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


# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.grid()
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()
