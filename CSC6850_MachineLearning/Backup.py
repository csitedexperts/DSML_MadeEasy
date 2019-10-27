# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:04:37 2019

@author: mhossa12
"""



# Developing a K-NN (Nearest Neighbors) Classification Model

# Importing the libraries
import pandas as pd
import sklearn as sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Importing the dataset2
dataset2 = pd.read_csv('A1Q3RawData.csv')


Outlook_categotical = dataset2.iloc[:, 1].values
Temperature_categotical = dataset2.iloc[:, 2].values
Humidity_categotical = dataset2.iloc[:, 3].values
Wind_categotical = dataset2.iloc[:, 4].values
Play_categotical = dataset2.iloc[:, 5].values


print(Outlook_categotical)
print(Temperature_categotical)
print(Humidity_categotical)
print(Wind_categotical)
print(Play_categotical)

#Combinig weather and temp into single listof tuples
categorical_features=zip(Outlook_categotical,Temperature_categotical, Humidity_categotical, Wind_categotical)
print (categorical_features)

# Import LabelEncoder
#from sklearn.preprocessing import LabelEncoder
# Let's create an object of the class LabelEncoder
# Import LabelEncoder
from sklearn import preprocessing

#creating labelEncoders
le = preprocessing.LabelEncoder()

Outlook_encoded = le.fit_transform(Outlook_categotical)
Temperature_encoded = le.fit_transform(Temperature_categotical)
Humidity_encoded = le.fit_transform(Humidity_categotical)
Wind_encoded = le.fit_transform(Wind_categotical)
Play_encoded = le.fit_transform(Play_categotical)

print(Outlook_encoded)
print(Temperature_encoded)
print(Humidity_encoded)
print(Wind_encoded)
print(Play_encoded)

### https://blog.usejournal.com/zip-in-python-48cb4f70d013

zipped_features=zip(Outlook_encoded,Temperature_encoded, Humidity_encoded, Wind_encoded)
list_zipped_features = list(zipped_features)
print ("zipped_features", list_zipped_features)
print("Play_encoded: ", Play_encoded)


## Model Generation
## After splitting, you will generate a random forest model on the training set and perform prediction on test set features 

#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier model
model  = GaussianNB()

#Train the model using the training sets
model.fit(list_zipped_features, Play_encoded)

#Predict Output
Play_predicted= model.predict([[0, 0, 0, 2]]) # ???????
print ("Play Predicted Value:", Play_predicted)

