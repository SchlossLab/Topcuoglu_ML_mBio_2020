#/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

#
# Author: Begum Topcuoglu
# Date: 2018-09-26
#
# This script runs all the models on Baxter Dataset subset of onlt cancer and normal samples to predict diagnosis based on OTU data only. This script only evaluates generalization performance of the model.
#

############################# IMPORT MODULES ##################################
import matplotlib
matplotlib.use('Agg') #use Agg backend to be able to use Matplotlib in Flux
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
from sklearn.preprocessing import MinMaxScaler
from timeit import default_timer as timer
#################################################################################



############################# PRE-PROCESS DATA ##################################
# Import the module I wrote preprocess_data

# In this case we will only need the function process_multidata which preprocesses shared and subsampled mothur generated OTU table and the metadata.This function will give us OTUs and FIT as features, diagnosis as labels.

# If we wanted to use only OTUs and not FIT as a feature, import the function process_data and use that.
#################################################################################

from preprocess_data import process_SRNdata
shared = pd.read_table("data/baxter.0.03.subsample.shared")
meta = pd.read_table("data/metadata.tsv")

# Define x (features) and y (labels)
x, y = process_SRNdata(shared, meta)

# When we use process_multidata:
# x: all the OTUs and FIT as features
# y: labels which are diagnosis of patient (0 is for non-advanced adenomas+normal colon and 1 is for advanced adenomas+carcinomas)



############################ MODEL SELECTION ####################################
# Import the module I wrote model_selection and function select_model

# This function will define the cross-validation method, hyper-parameters to tune and the modeling method based on which models we want to use here.
#################################################################################

from model_selection import select_model

# Define the models you want to use
models = ["L2_Logistic_Regression", "L1_SVM_Linear_Kernel", "L2_SVM_Linear_Kernel", "SVM_RBF", "Random_Forest", "Decision_Tree", "XGBoost"]



############################ TRAINING THE MODEL ###############################
## We will split the dataset 80%-20% and tune hyper-parameter on the 80% training and choose a best model and best hyper-parameters. The chosen best model and hyper-parameters will be tested on the %20 test set that was not seen before during training. This will give a TEST AUC. This is repeated 100 times anf will give 100 TEST AUCs. We call this the outer cross validation/testing.

## To tune the hyper-parameter we also use an inner cross validation that splits to 80-20 and repeats for 100 times. We report Cross-Validation AUC for this inner cross-validation.

## Here we use a for loop to iterate each model method.
#################################################################################
walltimes = []
for models in models:
    start = timer()
    print(models)
    ## Generate empty lists to fill with AUC values for test-set
    tprs_test = []
    aucs_test = []
    mean_fpr_test = np.linspace(0, 1, 100)
    ## Generate empty lists to fill with AUC values for train-set cv
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    ## Generate empty lists to fill with hyper-parameter and mean AUC
    scores = []
    names = []
    ## Define how many times we will iterate the outer crossvalidation
    i=0
    epochs= 100
    for epoch in range(epochs):
        i=i+1
        print(i)
        ## Split dataset to 80% training 20% test sets.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify = y)
        sc = MinMaxScaler(feature_range=(0, 1))
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        ## Define which model, parameters we want to tune and their range, and also the inner cross validation method(n_splits, n_repeats)
        model, param_grid, cv = select_model(models)
        ## Based on the chosen model, create a grid to search for the optimal model
        grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv, scoring = 'roc_auc', n_jobs=-1)
        ## Get the grid results and fit to training set
        grid_result = grid.fit(x_train, y_train)
        ## Print out the best model chosen in the grid
        print('Best model:', grid_result.best_estimator_)
        ## Print out the best hyper-parameters chosen in the grid
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        ## Calculate the AUC means and standard deviation for each hyper-parameters used during tuning. Print this out.
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        # Save the AUC means for each tested hyper-parameter tuning. Append for each outer cross validaion (epoch)
        # We want to plot this to see our hyper-parameter tuning performance and budget
        scores.append(means)
        names.append(i)
        parameters=pd.DataFrame(params)
        ## The best model we pick here will be used for predicting test set.
        best_model = grid_result.best_estimator_
        ## variable assignment to make it easier to read.
        X=x_train
        Y=y_train
        ## Calculate the FPR and TPR at each inner-cross validation and append these data to plot ROC curve for cross-validation with n_splits=5 and n_repeats=100 to evaluate the variation of prediction in our training set.
        for train, test in cv.split(X,Y):
            if models=="L2_Logistic_Regression" or models=="Random_Forest" or models=="XGBoost" or models=="Decision_Tree":
                y_score = best_model.fit(X[train], Y[train]).predict_proba(X[test])
                fpr, tpr, thresholds = roc_curve(Y[test], y_score[:, 1])
            else:
                y_score = best_model.fit(X[train], Y[train]).decision_function(X[test])
                fpr, tpr, thresholds = roc_curve(Y[test], y_score)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            print("Train", roc_auc)
        ## Calculate the FPR and TPR at each outer-cross validation and append these data to plot ROC curve for testing during 100 repeats(epochs) to evaluate the variation of prediction in our testing set.
        if models=="L2_Logistic_Regression" or models=="Random_Forest" or models=="XGBoost" or models=="Decision_Tree":
            y_score = best_model.fit(x_train, y_train).predict_proba(x_test)
            # Compute ROC curve and area the curve
            fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_score[:, 1])
        else:
            y_score = best_model.fit(x_train, y_train).decision_function(x_test)
            # Compute ROC curve and area the curve
            fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_score)
        tprs_test.append(interp(mean_fpr_test, fpr_test, tpr_test))
        tprs_test[-1][0] = 0.0
        roc_auc_test = auc(fpr_test, tpr_test)
        aucs_test.append(roc_auc_test)
        print("Test", roc_auc_test)
    ## Plot the ROC curve for inner and outer cross-validation
    plt.plot([0, 1], [0, 1], linestyle='--', color='green', label='Random', alpha=.8)
    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_test[-1] = 1.0
    mean_auc_test = auc(mean_fpr_test, mean_tpr_test)
    std_auc_test = np.std(aucs_test)
    plt.plot(mean_fpr_test, mean_tpr_test, color='r', label=r'Never-before-seen test set ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_test, std_auc_test), lw=2, alpha=.8)
    std_tpr_test = np.std(tprs_test, axis=0)
    tprs_upper_test = np.minimum(mean_tpr_test + std_tpr_test, 1)
    tprs_lower_test = np.maximum(mean_tpr_test - std_tpr_test, 0)
    plt.fill_between(mean_fpr_test, tprs_lower_test, tprs_upper_test, color='tomato', alpha=.2, label=r'$\pm$ 1 std. dev.')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean cross-val ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for %s' % models)
    plt.legend(loc="lower right", fontsize=8)
    save_results_to = 'results/figures/all_samples/'
    plt.savefig(save_results_to + str(models) + ".png", format="PNG", dpi=1000)
    plt.clf()


    # Save the CV-auc and Test-auc lists to a dataframe and then to a tab-delimited file
    cv_aucs= {'AUC':aucs}
    cv_aucs_df= pd.DataFrame(cv_aucs)
    test_aucs = {'AUC':aucs_test}
    test_aucs_df = pd.DataFrame(test_aucs)
    concat_aucs_df = pd.concat([cv_aucs_df,test_aucs_df], axis=1, keys=['Cross-validation','Testing']).stack(0)
    concat_aucs_df = concat_aucs_df.reset_index(level=1)
    save_results_to = 'data/process/'
    concat_aucs_df.to_csv(save_results_to + str(models) + ".tsv", sep='\t')

    # Save the hyper-parameter tuning performance
    full=pd.DataFrame.from_items(zip(names,scores))
    full=parameters.join(full)
    full.to_csv(save_results_to + str(models) + "_parameters.tsv", sep='\t')

    # Time each model
    end = timer()
    print(end - start)
    walltime = end - start
    # Append with loop
    walltimes.append(walltime)

# Save to a .tsv file 0 -> Logistic 6 -> XGBoost
print_walltimes = pd.DataFrame(walltimes)
print_walltimes.to_csv(save_results_to + "walltimes.tsv", sep='\t')
