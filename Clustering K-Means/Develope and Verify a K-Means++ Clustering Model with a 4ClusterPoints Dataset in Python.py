# Developing a K-Means Clustering Model

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('4Clustered_Points.csv')
X = dataset.iloc[:, [0, 1]].values
# y = dataset.iloc[:, 3].values
X_ordinate = dataset.iloc[:, 0].values
Y_ordinate = dataset.iloc[:, 1].values

"""
Visualize the row dataset into two dimensional scatter splot.

"""
plt.scatter(X_ordinate,Y_ordinate)  
plt.title("Plot Visualization for the Raw data")
plt.xlabel("X_ordinate values")
plt.ylabel("Y_ordinate values")
plt.grid()
plt.show()


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Employee K-Means Clusters')
plt.xlabel('Annual Income ($)')
plt.ylabel('Bonus Points (1-100)')
plt.legend()
plt.grid()
plt.show()