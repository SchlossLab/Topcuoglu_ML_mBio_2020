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
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

def select_model(net):
    ## Define the n-folds for hyper-parameter optimization on training set.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=25, random_state=200889)
    if net=="L2_Logistic_Regression":
        model = linear_model.LogisticRegression()
        ## We will try these regularization strength coefficients to optimize our model
        ## We will try these regularization strength coefficients to optimize our model
        c_values = [0.01, 0.1, 1, 10]
        param_grid = dict(C=c_values)
    if net=="L2_MultiClass_Logistic_Regression":
        model = linear_model.LogisticRegression(multi_class='multinomial', solver="lbfgs")
        ## We will try these regularization strength coefficients to optimize our model
        ## We will try these regularization strength coefficients to optimize our model
        c_values = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
        param_grid = dict(C=c_values)
    if net=="L1_SVM_Linear_Kernel":
        model= LinearSVC(penalty='l1', loss='squared_hinge', dual=False)
        c_values = [0.001, 0.01, 0.1 ,1]
        param_grid = dict(C=c_values)
    if net=="L2_SVM_Linear_Kernel":
        model = SVC(kernel='linear')
        c_values = [0.01, 0.1, 1, 10]
        param_grid = dict(C=c_values)
    if net=="SVM_RBF":
        model = SVC(kernel='rbf')
        c_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
        gamma = [0.000000001, 0.00000001, 0.0000001]
        param_grid = dict(C=c_values, gamma=gamma)
    if net=="Random_Forest":
        model = RandomForestClassifier(bootstrap= True)
        n_estimators = [1000]
        max_features= [10, 80, 500, 1000, 1500]
        param_grid = dict(n_estimators=n_estimators, max_features=max_features)
    if net=="Decision_Tree":
        model = DecisionTreeClassifier()
        max_depth=[5, 10, 50]
        min_samples_split=[10, 25, 50]
        param_grid = dict(max_depth=max_depth, min_samples_split=min_samples_split)
    if net=="XGBoost":
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=200889)
        ind_params={
        'n_estimators':100,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic'
        }
        model = xgb.XGBClassifier(**ind_params)
        param_grid = {
        'learning_rate':[0.1],
        'subsample': [0.7,0.8,0.9],
        'max_depth':[6,7,8],
        'min_child_weight':[1,2,3]
        }
    return model, param_grid, cv
