
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

# Splitting the dataset4 into the Training set and Test set
#from sklearn import cross_validation as cv
## cross_validation is deprecated since version 0.18. This module will be removed in 0.20. Use sklearn.model_selection.train_test_split instead.
## Source: https://stackoverflow.com/questions/53978901/importerror-cannot-import-name-cross-validation-from-sklearn


from sklearn.model_selection import train_test_split


# Splitting the data sets into Training and Test sets

X_train, X_test = train_test_split(X, test_size = .2, random_state = 0)
y_train, y_test = train_test_split(y, train_size = .8, random_state = 0)


# Training  => 80% 
## Just keelping backup copies
X_train0, X_test0 = X_train, X_test
y_train0, y_test0 = y_train, y_test 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
# The Classifier codes go here
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
#classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
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
X_set, y_set = X_test, y_test
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
