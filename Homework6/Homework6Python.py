
"""
#########   Homework_6.1(a)  ################
"""   

# Developing a K-Means++ Clustering Model

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Importing the Breast dataset
BreastDS = pd.read_csv("breast_data.csv")
X0 = BreastDS.iloc[:, 0:30].values


# Applying the PCA Dimentionality Reuction model
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X0)
# Reduced the 30 dimension into 2 dimension

# Importing the Breast_Truth dataset
BreastTDS = pd.read_csv("breast_truth.csv")
T = BreastTDS.iloc[:, 0].values

# Importing the Mu_Init dataset
MuInitDS = pd.read_csv("mu_init.csv")
M = MuInitDS.iloc[:, 0:2].values

print (M[:, 0].size)
x1 = M[:, 0]
x2 = M[:, 1]
colors = ['b', 'g', 'c']
markers = ['o', 'v', 's']

K = 3
kmeans_model = KMeans(n_clusters=K).fit(X)

print(kmeans_model.cluster_centers_)

center_size = M[:, 0].size

centers = np.array(kmeans_model.cluster_centers_)

plt.plot()
plt.grid()
plt.title('k means centroids')

print (enumerate(kmeans_model.labels_))

for i, l in range (0, center_size ): # enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
    plt.xlim([0, 10])
    plt.ylim([0, 10])

plt.scatter(centers[:,0], centers[:,1], marker="*", color='r')
plt.grid()
plt.show()


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters with randomly generated centers

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Sample Breast Cancer Diagnostic Dataset Clusters")
plt.xlabel("First reduced dimension")
plt.ylabel("Second reduced dimension")
plt.legend()
plt.grid()
plt.show()

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Within Cluster Sum of Squires (WCSS)")
plt.grid()
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters with randomly generated centers

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Sample Breast Cancer Diagnostic Dataset Clusters")
plt.xlabel("First reduced dimension")
plt.ylabel("Second reduced dimension")
plt.legend()
plt.grid()
plt.show()
