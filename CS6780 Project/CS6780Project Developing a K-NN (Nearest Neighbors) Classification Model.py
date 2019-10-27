# Developing a K-NN (Nearest Neighbors) Classification Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X_Train, y_Train)

# Predicting the Test set results
y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_Test, y_Pred)

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
plt.title('K-NN Plot for the Training dataset)')
plt.xlabel('Education in Year')
plt.ylabel('Annual Salary')
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
plt.title('K-NN for the Test dataset)')
plt.xlabel('Education in Year')
plt.ylabel('Annual Salary')
plt.legend()
plt.grid()
plt.show()