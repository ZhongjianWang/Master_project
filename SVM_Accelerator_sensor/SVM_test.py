#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 07:29:40 2020

@author: jason
"""
#importing the library 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#preprocessing dataset
df = pd.read_csv('./WISDM_ar_v1.1/WISDM_ar_v1.1_raw_svm.txt')
df.columns = ['accelerator']
df[['labels','0','1','2','3']] = df.accelerator.str.split(' ',expand = True)
df['mean'] = df['0'].apply(lambda x: x.split('0:')[1])
df['max'] = df['1'].apply(lambda x: x.split('1:')[1])
df['min'] = df['2'].apply(lambda x: x.split('2:')[1])
df['std'] = df['3'].apply(lambda x: x.split('3:')[1])

df_svm =df[['mean','max','min','std','labels']]
df_svm.to_csv('accelerator_cleaned.csv',index=False)



#importing the dataset
X= df_svm.iloc[:,:-1].values
y= df_svm.iloc[:,-1].values

#spliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

#traning model on the traning set 
from sklearn.svm import SVC
classifier = SVC(C=8192.0,kernel='rbf',gamma=0.03125)
classifier.fit(X_train, y_train)

#predicting a single new result 
#print(classifier.predict(sc.transform([[12,23,2,6]])))

#predicting the test set result
y_pred= classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


#making the confusion matrix 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# Generate confusion matrix
from sklearn.metrics import plot_confusion_matrix
#from mlxtend.plotting import plot_decision_regions
matrix = plot_confusion_matrix(classifier, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show(matrix)
plt.show()