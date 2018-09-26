#!/usr/bin/python
#
# Author: Begum Topcuoglu
# Date: 2018-09-26
#
# This script runs Logistic Regression analysis on Baxter Dataset subset of onlt cancer and normal samples to predict diagnosis based on OTU data only. This script only evaluates generalization performance of the model.
#

############## IMPORT MODULES ######################
import load_modules
import_modules()
############## PRE-PROCESS DATA ######################
import preprocess_data
process_data()
################## Logistic Regression ###############

## We will split the dataset 80%-20% and tune hyper-parameter on the 80% training. This will be done 100 times wth 5 folds and an optimal hyper-parameter/optimal model will be chosen.

## The chosen best model and hyper-parameter will be tested on the %20 test set that was not seen before during training. This will give a TEST AUC.

## We will split and redo previous steps 100 epochs. Which means we have 100 models that we test on the 20%. We will report the mean TEST AUC +/- sd.

# For each epoch, we will also report mean AUC values +/- sd for each cross-validation during training.

Logit_plot = plt.figure()
i=0
epochs= 1
for epoch in range(epochs):
    i=i+1
    print(i)
    ## Split dataset to 80% training 20% test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    y_train=y_train.values
    ## Define L2 regularized logistic classifier
    logreg = linear_model.LogisticRegression()
    ## Define the n-folds for hyper-parameter optimization on training set.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=200889)
    ## We will try these regularization strength coefficients to optimize our model
    C = {"C": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]}
    ## Define the best model:
    grid = GridSearchCV(logreg, C, cv=cv, verbose=1, scoring='roc_auc', n_jobs=-1)
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
    ## Generate empty lists to fill with AUC values for train-set cv
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
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

plt.plot([0, 1], [0, 1], linestyle='--', color='green', label='Luck', alpha=.8)
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
plt.title('L2 Logistic Regression ROC\n')
plt.legend(loc="lower right", fontsize=8)
#plt.show()
Logit_plot.savefig('results/figures/Logit_Baxter.png', dpi=1000)
