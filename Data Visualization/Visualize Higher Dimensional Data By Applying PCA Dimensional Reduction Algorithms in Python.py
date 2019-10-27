
# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset from the data file
datasetDS = pd.read_csv('CompSalary_Data.csv')

X = datasetDS.iloc[:, 0:7].values ## NOT  X = datasetDS.iloc[:, 1].values
## Since X is a Matrix here
y = datasetDS.iloc[:, 7:8].values  ## NOT  y = datasetDS.iloc[:, 7].values
## Since y is a Transposed Matrix of the y itself

#print (X)
#print (y)

## Just keelping backup copies
X0, y0 = X, y
#print (X0)
#print (y0)


### Could we do this ??



# Visualize the Original dataset
"""
plt.scatter(X, y, color = 'blue')
plt.plot(X, y, color='red')
plt.title("Original Set's Experience vs Salary Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Yearly Salary ($)")
plt.grid()
plt.show()

"""








### No, this is not a 2D matrix 





















# Applying the PCA Dimentionality Reuction model
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
X1 = pca.fit_transform(X)


# Visualize the Condenced dataset

plt.scatter(X1, y, color = 'blue')
plt.plot(X1, y, color='red')
plt.title("Original Set's Experience vs Salary Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Yearly Salary ($)")
plt.grid()
plt.show()





# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 1, kernel = 'rbf')
X3 = kpca.fit_transform(X)

# Visualize the Condenced dataset

plt.scatter(X3, y, color = 'blue')
plt.plot(X3, y, color='red')
plt.title("Original Set's Experience vs Salary Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Yearly Salary ($)")
plt.grid()
plt.show()


"""
# LDA NOT applicable here ????

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 1)
X2 = lda.fit_transform(X, y+1)

# Visualize the Condenced dataset

plt.scatter(X2, y, color = 'blue')
plt.plot(X2, y, color='red')
plt.title("Original Set's Experience vs Salary Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Yearly Salary ($)")
plt.grid()
plt.show()


"""
