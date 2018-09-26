#!/usr/bin/python
#
# Author: Begum Topcuoglu
# Date: 2018-09-26
#
# This script chooses the model parameters for each model selection
#



import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
import xgboost as xgb

def select_model(net):
    if net=="logreg"
    model = linear_model.LogisticRegression()
    ## We will try these regularization strength coefficients to optimize our model
    ## We will try these regularization strength coefficients to optimize our model
    c_values = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
    param_grid = dict(C=c_values)
    return model, param_grid
