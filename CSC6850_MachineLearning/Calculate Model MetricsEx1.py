# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:21:15 2019

@author: mhossa12
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


dataset122 = pd.read_csv('A1Q2Data.csv')
X = dataset122.iloc[:, 1:5].values
y = dataset122.iloc[:, 5].values

print (X)
print (y)



from sklearn.model_selection import train_test_split

# Splitting the data sets into Training and Test sets

X_train, X_test = train_test_split(X, test_size = .2, train_size = .8, random_state = 0)
y_train, y_test = train_test_split(y, test_size = .2, train_size = .8, random_state = 0)

print (X)
print (X_train)
print (X_test)



#from sklearn.metrics import recall_score

### Calculating Model Metrices using GaussianNB Classifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


gnb = GaussianNB()
#classifier = KNeighborsClassifier(n_neighbors = 6)
gnb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = gnb.predict(X_test)
gnb.score(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames = ["Actual"], colnames = ["Predicted"], margins = True)

classifi_report = classification_report(y_test, y_pred)
print(classifi_report)




y_pred_prob = gnb.predict_proba(X_test)[:,1]
print (y_pred_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('Gaussian Naive-Bayes (50/50) ROC Curve')
plt.grid()
plt.show()

#####################

""""
Alternative Way: Using KNN


"""


print (roc_auc_score(y_test, y_pred_prob))

### Calculating Model Metrices using KNeighborsClassifier Classifier 

from sklearn.neighbors import KNeighborsClassifier as knc

classifier = knc(n_neighbors = 3, metric = 'minkowski', p = 2)
#classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

recall_score(y_test, y_pred, average='macro')  
recall_score(y_test, y_pred, average='micro')  

recall_score(y_test, y_pred, average='weighted')  


print('Accuracy Score', accuracy_score(y_test, y_pred)  )
print('Precision Score', precision_score(y_test, y_pred)  )
print('Recall Score', recall_score(y_test, y_pred)  )
print('F1 Score', f1_score(y_test, y_pred)  )



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

y_pred = cm.predict(X_test)

matrix.fit(X_train, y_train)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
print (y_pred)
