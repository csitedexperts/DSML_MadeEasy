# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:17:48 2019

@author: mhossa12
"""


"""
#########   Assignment1_2  ################
2.1 Develop a Naive Bayes Classification Model to Predict a Label
2.2 Calculate the accuracy of the classifier with precision, recall, F1-score, sensitivity, specificity and ROC curve
 using the given dataset
 
"""


# Developing a Naive Bayes Classification Model to Predict a Label


# Importing the libraries
import math
import pandas as pd
import sklearn as sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Importing the dataset12
dataset12 = pd.read_csv('A1Q2RawData.csv')


Outlook_categotical = dataset12.iloc[:, 1].values
Temperature_categotical = dataset12.iloc[:, 2].values
Humidity_categotical = dataset12.iloc[:, 3].values
Wind_categotical = dataset12.iloc[:, 4].values
Play_categotical = dataset12.iloc[:, 5].values


print(Outlook_categotical)
print(Temperature_categotical)
print(Humidity_categotical)
print(Wind_categotical)
print(Play_categotical)

#Combinig Categorical features into a single list of tuples using zip()
zipped_categorical_features=zip(Outlook_categotical,Temperature_categotical, Humidity_categotical, Wind_categotical)
print (zipped_categorical_features)

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

#Combinig encoded features into a single list of tuples using zip()

zipped_encoded_features=zip(Outlook_encoded,Temperature_encoded, Humidity_encoded, Wind_encoded)
zipped_encoded_features_list = list(zipped_encoded_features)
print ("Zipped encoded features list", zipped_encoded_features_list)
print("Encoded Play Code [1-> Play, 0 -> Not Play]: ", Play_encoded)


## Naive Bayes Model Generation

#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier model
model  = GaussianNB()

#Train the model using the training sets
model.fit(zipped_encoded_features_list, Play_encoded)

print ("Zipped features list", zipped_encoded_features_list)
print("Encoded Play Code [1-> Play, 0 -> Not Play]: ", Play_encoded)

#Predict an Output for a combination of the features
feature_combination = [[2, 1, 1, 1]]  # ???????
Play_predicted= model.predict(feature_combination) 
print ("For the feature combination ", feature_combination, " Predicted Play Label : ", Play_predicted)

#Predict an Output for a combination of the features
feature_combination = [[2, 1, 0, 0]]  # ???????
Play_predicted= model.predict(feature_combination) 
print ("For the feature combination ", feature_combination, " Predicted Play Label : ", Play_predicted)

#Predict an Output for a combination of the features
feature_combination = [[0, 1, 0, 1]]  # ???????
Play_predicted= model.predict(feature_combination) 
print ("For the feature combination ", feature_combination, " Predicted Play Label : ", Play_predicted)

#Predict an Output for a combination of the features
feature_combination = [[2, 0, 0, 1]]  # ???????
Play_predicted= model.predict(feature_combination) 
print ("For the feature combination ", feature_combination, " Predicted Play Label : ", Play_predicted)

