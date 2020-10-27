#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:36:44 2020

@author: jason
"""
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot

#importing the dataset
df = pd.read_csv('accelerator_cleaned.csv')

#gee the dataset 
def get_dataset():
    X= df.iloc[:,:-1].values
    y= df.iloc[:,-1].values
    return X, y

# get a voting ensemble of models
def get_voting():
	# define the base models
	models = list()
	models.append(('svm1', SVC(probability=True,kernel='rbf',C=8192.0,gamma=0.03125, degree=1)))
	models.append(('svm2', SVC(probability=True,kernel='rbf',C=8192.0,gamma=0.03125, degree=2)))
	
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='soft')
	return ensemble
 
# get a list of models to evaluate
def get_models():
	models = dict()
	models['svm1'] = SVC(probability=True,kernel='rbf',C=8192.0,gamma=0.03125, degree=1)
	models['svm2'] = SVC(probability=True,kernel='rbf',C=8192.0,gamma=0.03125, degree=2)
	models['soft_voting'] = get_voting()
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores
 
# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()