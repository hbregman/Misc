# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 09:56:51 2016

@author: halliebregman
"""

#import libraries
import pandas as pd
import numpy as np
import os as os
import datetime as dt
import matplotlib.pyplot as plt
import time
%matplotlib inline
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import binarize
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

#Set working directory
os.chdir("/Home")
#Read in data
data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")

###Classification

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data1, x_vars=["x1", "x2", "x3", "x4"], 
             y_vars='y', size=7, aspect=0.7, hue='y')
sns.pairplot(data1, x_vars=["x5", "x6", "x7", "x8"], 
             y_vars='y', size=7, aspect=0.7, hue='y')
sns.pairplot(data1, x_vars=["x9", "x10", "x11", "x12"], 
             y_vars='y', size=7, aspect=0.7, hue='y')
       
       
####K Nearest Neighbors       
X = data1[['x1']]
y = data1[['y']]
y = np.squeeze(y)

knn = KNeighborsClassifier()
# define the parameter values that should be searched
k_range = range(5, 35)
weight_options = ['uniform', 'distance']
leaf_range = range(1, 50)

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options, leaf_size= leaf_range)

# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
best_scores = []
best_params = []
best_est = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_iter=10)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
    best_params.append(rand.best_params_)
    best_est.append(rand.best_estimator_)
    
print best_scores
print best_params
print best_est

###Using best parameters, apply to training/testing modelsß

X_train = data1[['x1']]
y_train = data1['y']
X_test = data2[['x1']]
y_test = data2['y']

# instantiate the model ()
knn = KNeighborsClassifier(n_neighbors=14, weights='uniform', leaf_size=31)

# fit the model with data
knn.fit(X_train, y_train)

# predict the response for new observations
y_pred = knn.predict(X_test)

confusion = metrics.confusion_matrix(y_test, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[1, 0]
FN = confusion[0, 1]

#Confusion matrix
print confusion
#Classification accuracy
print metrics.accuracy_score(y_test, y_pred)
#Classification error
print 1 - metrics.accuracy_score(y_test, y_pred)
#Sensitivity
print metrics.recall_score(y_test, y_pred)
#Specificity
print TN / float(TN + FP)
#False positive rate
print FP / float(TN + FP)
#Precision
print TP / float(TP + FP)

###Cross validation score of accuracy
print cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy').mean()


####Random Forest
X = binder2830[['totV1']]
y = binder2830[['rain_carl']]
y = np.squeeze(y)

knn = RandomForestClassifier()
# define the parameter values that should be searched
k_range = range(5, 35)
weight_options = ['auto', 'sqrt', 'log2']
leaf_range = range(1, 50)

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_estimators=k_range, max_features=weight_options, max_depth= leaf_range)

# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
best_scores = []
best_params = []
best_est = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_iter=10)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
    best_params.append(rand.best_params_)
    best_est.append(rand.best_estimator_)
    
print best_scores
print best_params
print best_est

X_train = data1[['x1']]
y_train = data1['y']
X_test = data2[['x1']]
y_test = data2['y']

# instantiate the model ()
knn = RandomForestClassifier(max_depth=2, n_estimators=9, max_features='sqrt')

# fit the model with data
knn.fit(X_train, y_train)

# predict the response for new observations
y_pred = knn.predict(X_test)
#y_pred[X_test.totV1 == 0] = 1

confusion = metrics.confusion_matrix(y_test, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[1, 0]
FN = confusion[0, 1]

#Confusion matrix
print confusion
#Classification accuracy
print metrics.accuracy_score(y_test, y_pred)
#Classification error
print 1 - metrics.accuracy_score(y_test, y_pred)
#Sensitivity
print metrics.recall_score(y_test, y_pred)
#Specificity
print TN / float(TN + FP)
#False positive rate
print FP / float(TN + FP)
#Precision
print TP / float(TP + FP)

###Cross validattion score of accuracy
print cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()

# print the first 10 predicted probabilities of class membership
print knn.predict_proba(X_test)[0:10, :]

###Regression

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data1, x_vars=["x1", "x2", "x3", "x4"], 
             y_vars='y2', size=7, aspect=0.7, hue='y2')
sns.pairplot(data1, x_vars=["x5", "x6", "x7", "x8"], 
             y_vars='y2', size=7, aspect=0.7, hue='y2')
sns.pairplot(data1, x_vars=["x9", "x10", "x11", "x12"], 
             y_vars='y2', size=7, aspect=0.7, hue='y2')
             
X = data1[['x2']]
y = data1[['y2']]
y = np.squeeze(y)

# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# instantiate the model 
knn = LinearRegression()

# fit the model with data
knn.fit(X_train, y_train)

# predict the response for new observations
y_pred = knn.predict(X_test)
#y_pred[X_test.totV1 == 0] = 1

###Average RMSE for cross validated data
print np.sqrt(-cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')).mean()

#Create dataframe with coefficients
coefs = pd.DataFrame(zip(X.columns, knn.coef_), columns=['features', 'estimatedCoefficients'])
print coefs

#Plot feature importance
sns.stripplot(x="features", y="estimatedCoefficients", data=coefs);

###Set datetime as index
times = pd.DatetimeIndex(data2["datetime"])
data2['newday'] = times.day #set new variable equal to day
data2['hour'] = times.hour #set hour to hour
data2['day'] = times.day #set day to day
data2.ix[data2.hour < 7, 'newday'] = data2['day']-1 #based on hour, redefine day
grouped = data2.groupby([times.year, times.month, 'newday']) #aggregate to newday
data2_newday = grouped.agg({'y_pred' : np.sum #sum y_pred
                            })
data2_newday = data2_newday.reset_index() #reset i ndex
data2_newday.columns = ('year', 'month', 'day', 'pred_y') #rename columns
data2_newday #evaluate values aggegated to "newday"           
             
X_train = data1[['x2', 'x3']]
y_train = data1['y2']
X_test = data2[['x2', 'x3']]
y_test = data2['y2']

# instantiate the model 
knn = GradientBoostingRegressor(n_estimators=9)
# fit the model with data
knn.fit(X_train, y_train)

# predict the response for new observations
y_pred = knn.predict(X_test)
#y_pred[X_test.totV1 == 0] = 1

print np.sqrt(-cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')).mean()

coefs = pd.DataFrame(zip(X.columns, knn.feature_importances_), columns=['features', 'importance'])
print coefs.sort("importance")

sns.stripplot(x="features", y="importance", data=coefs.sort("importance"));

data2['y_pred'] = pd.Series(y_pred)            

###Set datetime as index
times = pd.DatetimeIndex(data2["datetime"])
data2['newday'] = times.day #set new variable equal to day
data2['hour'] = times.hour #set hour to hour
data2['day'] = times.day #set day to day
data2.ix[data2.hour < 7, 'newday'] = data2['day']-1 #based on hour, redefine day
grouped = data2.groupby([times.year, times.month, 'newday']) #aggregate to newday
data2_newday = grouped.agg({'y_pred' : np.sum #sum y_pred
                            })
data2_newday = data2_newday.reset_index() #reset i ndex
data2_newday.columns = ('year', 'month', 'day', 'pred_y') #rename columns
data2_newday #evaluate values aggegated to "newday"           