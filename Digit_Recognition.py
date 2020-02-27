
#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#read train and test files
train_file = pd.read_csv('train.csv')

#list of all digits that are going to be predicted
np.sort(train_file.label.unique())

#define the number of samples for training set and for validation set
num_train,num_validation = int(len(train_file)*0.8),int(len(train_file)*0.2)
Print(num_train,num_validation)

#generate training data from train_file
x_train,y_train=train_file.iloc[:num_train,1:].values,train_file.iloc[:num_train,0].values
x_validation,y_validation=train_file.iloc[num_train:,1:].values,train_file.iloc[num_train:,0].values

print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)

#fit a Random Forest classifier
clf=RandomForestClassifier(n_estimators=10,criterion='10',random_state=0)
clf.fit(x_train,y_train)


#predict value of label using classifier
prediction_validation = clf.predict(x_validation)

print("Validation Accuracy: " + str(accuracy_score(y_validation,prediction_validation)))

print("Validation Confusion Matrix: \n" + str(confusion_matrix(y_validation,prediction_validation)))