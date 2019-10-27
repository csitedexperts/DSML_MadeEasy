
"""
#########   Assignment1_2  ################
2.1 Develop a Naive Bayes Classification Model to Predict a Label
2.2 Calculate the accuracy of the classifier with precision, recall, F1-score, sensitivity, specificity and ROC curve
 using the given dataset
 
"""




# Developing a Naive Bayes Classification Model to Predict a Label
# Importing the libraries


import os

import math
import pandas as pd
import sklearn as sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, classification_report

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
#print (zipped_categorical_features)

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

print(Outlook_categotical)
print(Temperature_categotical)
print(Humidity_categotical)
print(Wind_categotical)
print(Play_categotical)
print(Outlook_categotical)

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



#Predict an Output for a combination of the features
feature_combination = [[2, 2, 1, 1]]  # ???????
Play_predicted= model.predict(feature_combination) 
print ("For the feature combination ", feature_combination, " Predicted Play Label : ", Play_predicted)


#Predict an Output for a combination of the features
feature_combination = [[0, 0, 0, 0]]  # ???????
Play_predicted= model.predict(feature_combination) 
print ("For the feature combination ", feature_combination, " Predicted Play Label : ", Play_predicted)

""""
2.2 Calculate the accuracy of the classifier with precision, recall, F1-score, sensitivity, specificity and ROC curve
 using the given dataset
 
## Resource: https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
  
"""


# Splitting the dataset4 into the Training set and Test set

X = np.array(zipped_encoded_features_list )
y = Play_encoded


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



"""
#########   Assignment1_3  ################
3.1 Predict the output with KNN- Classifier Model 
3.2 Calculating the accuracy of the classifier with precision, recall, F1-score, sensitivity, specificity and ROC curve
 using the given dataset
 
"""




# Developing a K-NN Classification Model to Predict a Label Output


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

# Importing the dataset13
dataset13 = pd.read_csv('A1Q3RawData.csv')



X = dataset13.iloc[:, [1, 4]].values
y = dataset13.iloc[:, 5].values

## Since the dataset13 contains numerical values for the G1, G2, .., G7 features so no encoding needed for the features
## But Encoding is needed for the Output label - sicne it is Categorical Data (Yes/NO) form

G1_numerical = dataset13.iloc[:, 1].values
G2_numerical = dataset13.iloc[:, 2].values
G3_numerical = dataset13.iloc[:, 3].values
G4_numerical = dataset13.iloc[:, 4].values
G5_numerical = dataset13.iloc[:, 5].values
G6_numerical = dataset13.iloc[:, 6].values
G7_numerical = dataset13.iloc[:, 7].values
## Output label is Categorical
Output_categotical = dataset13.iloc[:, 8].values

# Import LabelEncoder
#from sklearn.preprocessing import LabelEncoder
# Let's create an object of the class LabelEncoder
# Import LabelEncoder
from sklearn import preprocessing

#creating labelEncoders
le = preprocessing.LabelEncoder()

Output_encoded = le.fit_transform(Output_categotical)

print (G1_numerical)
print (G2_numerical)
print (G3_numerical)
print (G4_numerical)
print (G5_numerical)
print (G6_numerical)
print (G7_numerical)
print ("Output Categorical ", Output_categotical)
print ("Output Encoded ", Output_encoded)

#Combinig/Zipping the features into a single list of tuples using zip() method

zipped_features=zip(G1_numerical, G2_numerical, G3_numerical, G4_numerical, G5_numerical, G6_numerical, G6_numerical)
zipped_features_list = list(zipped_features)
print ("Zipped features list", zipped_features_list)
print("Encoded Output Code [1-> Yes, 0 -> No]: ", Output_encoded)

#########  Verification with KNN Model  #############
#Create a KNN Classifier model

from sklearn.neighbors import KNeighborsClassifier as knc

model = knc(n_neighbors = 3)


#Train the model using the training sets
model.fit(zipped_features_list, Output_encoded)

print ("Zipped features list", zipped_features_list)
print("Encoded Output [1-> yes, 0 -> No]: ", Output_encoded)

#Predict an Output for a combination of the features
feature_combination = [[2.1, 2.2, 3.2, 1.4, 5.1, 2.4, 1.4]]  # ???????
Play_predicted= model.predict(feature_combination) 
print ("For the feature combination ", feature_combination, " Predicted Play Label : ", Play_predicted)


#Predict an Output for a combination of the features
feature_combination = [[2.4, 2.3, 3.4, 3.8, 2.3, 5.7, 5.2]]  # ???????
Play_predicted= model.predict(feature_combination) 
print ("For the feature combination ", feature_combination, " Predicted Play Label : ", Play_predicted)

################################

"""
If use Gaussian Distribution

model  = GaussianNB()
model.fit(X_train, y_train)


"""

"""
Q-3.2:  Consider the following example and calculate the accuracy of the classifier with precision, recall, F1-score, specificity and ROC curve using Python.  

"""


# Splitting the dataset4 into the Training set and Test set

X = np.array(zipped_encoded_features_list )
y = Play_encoded

# Splitting the data sets into Training and Test sets

X_train, X_test = train_test_split(X, test_size = .2, train_size = .8, random_state = 0)
y_train, y_test = train_test_split(y, test_size = .2, train_size = .8, random_state = 0)

print (X)
print (X_train)
print (X_test)


### Calculating Model Metrices using KNeighborsClassifier Classifier 

from sklearn.neighbors import KNeighborsClassifier as knc

classifier = knc(n_neighbors = 3)

classifier.fit(X_train, y_train)

print('Train Accuracy Score:', round(classifier.score(X_train, y_train), 2))

print('Test Accuracy Score:', classifier.score(X_test, y_test))


# Predicting the Test set results
y_pred = classifier.predict(X_test)


print('Accuracy Score', accuracy_score(y_test, y_pred)  )
print('Precision Score', precision_score(y_test, y_pred)  )
print('Recall Score', recall_score(y_test, y_pred)  )
print('F1 Score', f1_score(y_test, y_pred)  )


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

y_pred_prob = gnb.predict_proba(X_test)[:,1]
print (y_pred_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.title("KNN (K=3) ROC Curve")
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.show()



"""
#########   Assignment1_4  ################
4.1 Defining the Entropy and Gain functions in order to Generate Decision Tree 
using ID3 Algorithm

"""

"""

Entropy (One Attribute) [ E(S) ]

E(S)=∑ci=1−pilog2pi 
Entropy (Multi Attributes) [ E(T,X) ]

E(T,X)=∑c∈XP(c)E(c)=P(X1)⋅c1log2c1+P(X2)⋅c2log2c2+...+P(Xn)⋅cnlog2cn 
Information Gain [ G(T,X) ]

G(T,X)=E(T)−E(T,X) 
Best Split Point [ SplitInfoA(D) ]

SplitInfoA(D)=∑vj=1|Dj||D|⋅log2(|Dj||D|) 
Gain Ratio [ GainRatio(A) ]

GainRatio(A)=Gain(A)SplitInfo(A)


"""
## Let's define the required Entropy functions, as follows:


def Entropy1(x1):
    if x1 == 0:
        return 0
    else:
        entropy = -(x1*math.log(x1, 2))
    return entropy

def Entropy2(x1, x2):
    if x1 == 0 or x2 == 0:
        return 0
    else:
        entropy = -(x1*math.log(x1, 2) + x2*math.log(x2, 2))
    return entropy


def Entropy3(x1, x2, x3):
    if x1 == 0 or x2 == 0 or x3 ==0:
        return 0
    else:
        entropy = -(x1*math.log(x1, 2) + x2*math.log(x2, 2) + x3*math.log(x3, 2))
    return entropy


print("Entropy1(2/3) = ", Entropy1(2/3))
print("Entropy1(6/9) = ", Entropy1(6/9))

print("Entropy2(5, 9) = ", Entropy2(5, 9))

print("Entropy2(2/5, 3/5) = ", Entropy2(2/5, 3/5))
print("Entropy2(2/5, 1/5) = ", Entropy2(2/5, 1/5))
print("Entropy2(4/5, 3/5) = ", Entropy2(4/5, 3/5))
print("Entropy2(1/5, 3/5) = ", Entropy2(1/5, 3/5))
print("Entropy2(4/5, 2/5) = ", Entropy2(4/5, 2/5))
print("Entropy2(1/3, 2/3) = ", Entropy2(1/3, 2/3))


print("Entropy2(4/6, 2/6) = ", Entropy2(4/6, 2/6))
print("Entropy2(1/4, 2/2) = ", Entropy2(1/4, 2/2))

print("Entropy2(1, 1/3) = ", Entropy2(1, 1/3))

print("Entropy2(1/4, 3/4) = ", Entropy2(1/4, 3/4))

print("Entropy2(1, 1/3) = ", Entropy2(1, 1/3))

print("Entropy2(1/5, 0/5) = ", Entropy2(1/5, 0/5))

print("Entropy2(6/9, 3/9) = ", Entropy2(6/9, 3/9))

print("Entropy2(3/14, 2/14) = ", Entropy2(3/14, 2/14))
print("Entropy2(2/14, 3/14) = ", Entropy2(2/14, 3/14))

print("Entropy2(2, 3) = ", Entropy2(2, 3))


#print("Entropy2(4, 0) = ", Entropy2(4, 0))
print("Entropy2(5/14, 9/14) = ", Entropy2(5/14, 9/14))

print("Entropy2(6/9, 3/9) = ", Entropy2(6/9, 3/9))

print("== = ", ((3/9*Entropy2(2/9, 1/9) + (3/9)*Entropy2(1/9, 2/9))))

print("Entropy3(1/10, 3/10, 6/10) = ", Entropy3(1/10, 3/10, 6/10))


print("Entropy3(8/20, 10/20, 2/20) = ", Entropy3(8/20, 10/20, 2/20))

print("Gain(X, Y) = ", ( Entropy2(5/14, 9/14) - ((5/14)*Entropy2(3/14,2/14) + (5/14)*Entropy2(2/14,3/14)) ))



"""
#########   Assignment1_4  ################
4.1 Calculating the accuracy of the classifier with precision, recall, F1-score, sensitivity, specificity and ROC curve
 using the given dataset4
 
"""



from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


# Importing the dataset12
dataset14 = pd.read_csv('A1Q4RawData.csv')


Outlook_categotical = dataset14.iloc[:, 1].values
Temperature_categotical = dataset14.iloc[:, 2].values
Humidity_categotical = dataset14.iloc[:, 3].values
Wind_categotical = dataset14.iloc[:, 4].values
Play_categotical = dataset14.iloc[:, 5].values


print(Outlook_categotical)
print(Temperature_categotical)
print(Humidity_categotical)
print(Wind_categotical)
print(Play_categotical)


#Combinig Categorical features into a single list of tuples using zip()
zipped_categorical_features=zip(Outlook_categotical,Temperature_categotical, Humidity_categotical, Wind_categotical)
#print (zipped_categorical_features)

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


## Decision Tree Model Generation


X = np.array(zipped_encoded_features_list )
y = Play_encoded


from sklearn.model_selection import train_test_split

# Splitting the data sets into Training and Test sets

X_train, X_test = train_test_split(X, test_size = .2, train_size = .8, random_state = 0)
y_train, y_test = train_test_split(y, test_size = .2, train_size = .8, random_state = 0)

print (X)
print (X_train)
print (X_test)



### Calculating Model Metrices using KNeighborsClassifier Classifier 

#Create a DecisionTree Classifier model
dct_model  = DecisionTreeClassifier()

#Train the model using the training sets
dct_model.fit(zipped_encoded_features_list, Play_encoded)

print ("Zipped features list", zipped_encoded_features_list)
print("Encoded Play Code [1-> Play, 0 -> Not Play]: ", Play_encoded)



features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
dot_data = StringIO()
export_graphviz(dct_model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['No','Yes'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('dt_images/play_67-33.png')
Image(graph.create_png())


dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc = dtc.fit(X_train, y_train)
print('Train Accuracy Score:', dtc.score(X_train, y_train))
print('Test Accuracy Score:', dtc.score(X_test, y_test))


# Predicting the Test set results
y_pred = dct_model.predict(X_test)

confusion_matrix(y_test, y_pred)


features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
dot_data = StringIO()
export_graphviz(dtc, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['No','Yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('dt_images/treeIV_67-33.png')
Image(graph.create_png())



pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predicted'], margins=True)

print('Accuracy Score', accuracy_score(y_test, y_pred)  )
print('Precision Score', precision_score(y_test, y_pred)  )
print('Recall Score', recall_score(y_test, y_pred)  )
print('F1 Score', f1_score(y_test, y_pred)  )



y_pred_prob = gnb.predict_proba(X_test)[:,1]
print (y_pred_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.title("DT ROC Curve")
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.show()


