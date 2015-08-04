#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

# Subset the dataset
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print clf.score(features_test, labels_test)

# Output the specific answer
# answer = [pred[10],pred[26],pred[50]]
# print answer

# Count how many features are classified as Chris
count = 0
for i in pred:
    if i == 1:
        count += 1
print count


#########################################################

import math
entropy = -(2/3)*math.log(2/3, 2) - (1/3)*math.log(1/3, 2)
print entropy



