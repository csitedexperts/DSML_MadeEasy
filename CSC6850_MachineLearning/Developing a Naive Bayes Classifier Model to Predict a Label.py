
"""
#########   Assignment_2  ################
2.1 Develop a Naive Bayes Classification Model to Predict a Label
2.2 Calculate the accuracy of the classifier with precision, recall, F1-score, sensitivity, specificity and ROC curve
 using the given dataset1
 
"""


# Developing a Naive Bayes Classification Model to Predict a Label


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
dataset2 = pd.read_csv('A1Q2RawData.csv')


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
zipped_features_list = list(zipped_features)
print ("Zipped features list", zipped_features_list)
print("Encoded Play Code [1-> Play, 0 -> Not Play]: ", Play_encoded)


## Model Generation
## After splitting, you will generate a random forest model on the training set and perform prediction on test set features 

#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier model
model  = GaussianNB()

#Train the model using the training sets
model.fit(zipped_features_list, Play_encoded)

print ("Zipped features list", zipped_features_list)
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


## 2.2 

def get_predictions(clf, X_train, y_train, X_test):
    # create classifier
    clf = clf
    # fit it to training data
    clf.fit(X_train,y_train)
    # predict using test data
    y_pred = clf.predict(X_test)
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    #for fun: train-set predictions
    train_pred = clf.predict(X_train)
    print('Train-set confusion matrix:\n', confusion_matrix(y_train,train_pred)) 
    return y_pred, y_pred_prob

def print_scores(y_test, y_pred, y_pred_prob):
    print('Test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    print("Recall score: ", recall_score(y_test,y_pred))
    print("Precision score: ", precision_score(y_test,y_pred))
    print("F1 score: ", f1_score(y_test,y_pred))
    print("Accuracy score: ", accuracy_score(y_test,y_pred))
    
# Now let's print the scores
y_pred, y_pred_prob = get_predictions(LogisticRegression(C = 0.01, penalty = 'l1')
                                      , X_train, y_train, X_test)
print_scores(y_test,y_pred, y_pred_prob)


