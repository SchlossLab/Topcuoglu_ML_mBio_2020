#!/usr/bin/python
#
# Author: Begum Topcuoglu
# Date: 2018-09-26
#
# This script chooses the model parameters for each model selection
#




from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold

def select_model(net):
    ## Define the n-folds for hyper-parameter optimization on training set.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=200889)
    if net=="L2 Logistic Regression":
        model = linear_model.LogisticRegression()
        ## We will try these regularization strength coefficients to optimize our model
        ## We will try these regularization strength coefficients to optimize our model
        c_values = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
        param_grid = dict(C=c_values)
    if net=="L1 SVM Linear Kernel":
        model= LinearSVC(penalty='l1', loss='squared_hinge', dual=False)
        c_values = [0.1 ,1, 5, 10, 20, 25, 30, 50, 100]
        param_grid = dict(C=c_values)
    if net=="L2 SVM Linear Kernel":
        model = SVC(kernel='linear')
        c_values = [0.0001, 0.001, 0.01, 0.1, 1]
        param_grid = dict(C=c_values)
    if net=="SVM RBF":
        model = SVC(kernel='rbf')
        c_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
        gamma = [0.000000001, 0.00000001, 0.0000001]
        param_grid = dict(C=c_values, gamma=gamma)
    if net=="Random Forest":
        model = RandomForestClassifier(bootstrap= True)
        n_estimators = [1000, 2000, 3000, 4000, 5000, 6000]
        param_grid = dict(n_estimators=n_estimators)
    return model, param_grid, cv
