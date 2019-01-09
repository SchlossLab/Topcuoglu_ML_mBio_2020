#!/usr/bin/python
#
# Author: Begum Topcuoglu
# Date: 2018-09-26
#
# This script chooses the model parameters for each model selection
#




############################# IMPORT MODULES ##################################
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
#################################################################################



def select_model(net):
    ## Define the n-folds for hyper-parameter optimization on training set.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=200889)
    ## With the below if statements, we will define which classifier will be used.
    ## We will also define the hyper-parameters to be tuned and their budget for each model.
    if net=="L2_Logistic_Regression":
        model = linear_model.LogisticRegression(random_state=200889)
        c_values = [0.01, 0.1, 0.25, 0.5, 0.8, 0.9, 1, 10]
        param_grid = dict(C=c_values)
    if net=="L1_SVM_Linear_Kernel":
        model= LinearSVC(penalty='l1', loss='squared_hinge', dual=False, random_state=200889)
        c_values = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1 ,1]
        param_grid = dict(C=c_values)
    if net=="L2_SVM_Linear_Kernel":
        model= LinearSVC(penalty='l2', loss='squared_hinge', dual=False, random_state=200889)
        c_values = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1 ,1]
        param_grid = dict(C=c_values)
    if net=="SVM_RBF":
        model = SVC(kernel='rbf', random_state=200889)
        c_values = [0.00001, 0.0001, 0.001, 0.01]
        gamma = [0.000000000001, 0.00000000001, 0.0000000001, 0.000000001]
        param_grid = dict(C=c_values, gamma=gamma)
    if net=="Random_Forest":
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=200889)
        model = RandomForestClassifier(bootstrap= True, random_state=200889)
        n_estimators = [1000]
        max_features= [10, 80, 500, 1000, 1500, 2000, 3000]
        param_grid = dict(n_estimators=n_estimators, max_features=max_features)
    if net=="Decision_Tree":
        model = DecisionTreeClassifier(random_state=200889)
        max_depth=[6, 8, 10, 50]
        min_samples_split=[10, 25, 50, 75, 100]
        param_grid = dict(max_depth=max_depth, min_samples_split=min_samples_split)
    if net=="XGBoost":
        # https://jessesw.com/XG-Boost/
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=200889)
        ind_params={
        'n_estimators':500,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic'
        }
        model = xgb.XGBClassifier(**ind_params)
        param_grid = {
        'learning_rate':[0.01, 0.05, 0.1],
        'subsample': [0.5, 0.6, 0.7,0.8],
        'max_depth':[6,7,8],
        'min_child_weight':[1]
        }
    return model, param_grid, cv




## Print out the parameters that are being optimized for each model as a dataframe and export as .csv.
## This file will be used to generate Table 2 in our manuscript.
models = ["L2_Logistic_Regression", "L1_SVM_Linear_Kernel", "L2_SVM_Linear_Kernel", "SVM_RBF", "Random_Forest", "Decision_Tree", "XGBoost"]
params = []
for models in models:
    model, param_grid, cv = select_model(models)
    param_grid = pd.DataFrame.from_dict(data=param_grid, orient='index')
    num= len(param_grid.columns)
    rng = range(1, num + 1)
    new_cols = [models + str(i) for i in rng]
    param_grid.columns = new_cols[:num]
    params.append(param_grid)

appended_params = pd.concat(params, axis=1)
appended_params.to_csv('data/process/param_grid.csv')
