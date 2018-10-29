#!/usr/bin/python
#
# Author: Begum Topcuoglu
# Date: 2018-09-26
#
# This script runs Logistic Regression analysis on Baxter Dataset subset of onlt cancer and normal samples to predict diagnosis based on OTU data only. This script only evaluates generalization performance of the model.
#

############## IMPORT MODULES ######################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sympy
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


############## PRE-PROCESS DATA ######################
from preprocess_data import process_data
from preprocess_data import process_multidata
shared = pd.read_table("data/baxter.0.03.subsample.shared")
meta = pd.read_table("data/metadata.tsv")
# Define x (features) and y (labels)
x, y = process_multidata(shared, meta)
################## MODEL SELECTION ###############
from model_selection import select_model

## We will split the dataset 80%-20% and tune hyper-parameter on the 80% training. This will be done 100 times wth 5 folds and an optimal hyper-parameter/optimal model will be chosen.

## The chosen best model and hyper-parameter will be tested on the %20 test set that was not seen before during training. This will give a TEST AUC.

## We will split and redo previous steps 100 epochs. Which means we have 100 models that we test on the 20%. We will report the mean TEST AUC +/- sd.

# For each epoch, we will also report mean AUC values +/- sd for each cross-validation during training.

## Generate empty lists to fill with AUC values for test-set
tprs_test = []
aucs_test = []
mean_fpr_test = np.linspace(0, 1, 100)
Logit_plot = plt.figure()
## Generate empty lists to fill with AUC values for train-set cv
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i=0
epochs= 25
for epoch in range(epochs):
    i=i+1
    print(i)
    ## Split dataset to 80% training 20% test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)
    ## Scale the dataset by removing mean and scaling to unit variance
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    ## Define which model, parameters we want to tune and their range, and also the cross validation method(n_splits, n_repeats)
    model, param_grid, cv = select_model("L2_Logistic_Regression")
    ## Based on the chosen model, create a grid to search for the optimal model
    grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv, scoring="roc_auc", n_jobs=1)
    ## Get the grid results and fit to training set
    grid_result = grid.fit(x_train, y_train)
    print('Best C:', grid_result.best_estimator_.get_params()['C'])
    print('Best model:', grid_result.best_estimator_)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    ## The best model we pick here will be used for predicting test set.
    best_model = grid_result.best_estimator_
    ## variable assignment to make it easier to read.
    X=x_train
    Y=y_train
    ## Plot mean ROC curve for cross-validation with n_splits=5 and n_repeats=100 to evaluate the variation of prediction in our training set.
    for train, test in cv.split(X,Y):
        probas_ = best_model.fit(X[train], Y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print("Train", roc_auc)
    ## Plot mean ROC curve for 100 epochs test set evaulation.
    probas_ = best_model.predict_proba(x_test)
    # Compute ROC curve and area the curve
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, probas_[:, 1])
    tprs_test.append(interp(mean_fpr_test, fpr_test, tpr_test))
    tprs_test[-1][0] = 0.0
    roc_auc_test = auc(fpr_test, tpr_test)
    aucs_test.append(roc_auc_test)
    print("Test", roc_auc_test)

d1= {'AUC':aucs}
df1= pd.DataFrame(d1)

d2= {'AUC':aucs_test}
df2= pd.DataFrame(d2)

df3 = pd.concat([df1,df2], axis=1, keys=['Cross-validation','Testing']).stack(0)
df3 = df3.reset_index(level=1)
import seaborn as sns

dots= sns.swarmplot(x='level_1',y='AUC', data=df3)
sns.boxplot(x='level_1', y='AUC', data=df3, ax=dots, showcaps=False,boxprops={'facecolor':'None'})
plt.show()
