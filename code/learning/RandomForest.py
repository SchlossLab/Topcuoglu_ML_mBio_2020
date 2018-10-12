############## IMPORT MODULES ######################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import *
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


############## PRE-PROCESS DATA ######################

## Read in the data

## shared has our input features(OTUs)
shared = pd.read_table("data/baxter.0.03.subsample.shared")
## meta has the labels(diagnosis)
meta = pd.read_table("data/metadata.tsv")

## Check and visualize the data
meta.head()
shared.head()

## Remove unnecessary columns from meta and only keep label for classification(diagnosis) and the sample name
meta = meta[['sample','dx']]

## Rename the column name "Group" to match the "sample" in meta
shared = shared.rename(index=str, columns={"Group":"sample"})

## Merge the 2 datasets on sample
data=pd.merge(meta,shared,on=['sample'])

## Remove adenoma samples. We will do a 2 classification model, just for cancer and normal colon samples.
data= data[data.dx.str.contains("adenoma") == False]

## Drop all except OTU columns for x
x = data.drop(["sample", "dx", "numOtus", "label"], axis=1)

## Cancer =1 Normal =0
diagnosis = { "cancer":1, "normal":0}

## Generate y which only has diagnosis as 0 and 1
y = data["dx"].replace(diagnosis)
# y = np.eye(2, dtype='uint8')[y]

## Drop if NA elements
y.dropna()
x.dropna()

## Generate empty lists to fill with AUC values for test-set
tprs_test = []
aucs_test = []
mean_fpr_test = np.linspace(0, 1, 100)

################## Random Forest ###############

## We will split the dataset 80%-20% and tune hyper-parameter on the 80% training. This will be done 100 times wth 5 folds and an optimal hyper-parameter/optimal model will be chosen.

## The chosen best model and hyper-parameter will be tested on the %20 test set that was not seen before during training. This will give a TEST AUC.

## We will split and redo previous steps 100 epochs. Which means we have 100 models that we test on the 20%. We will report the mean TEST AUC +/- sd.

# For each epoch, we will also report mean AUC values +/- sd for each cross-validation during training.

RF_plot = plt.figure()
i=0
epochs= 10
for epoch in range(epochs):
    i=i+1
    print(i)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    y_train=y_train.values

# Decide on the number of decision trees
    #param_grid = {'n_estimators': [ 25, 50, 100, 120, 150, 300, 500, 800, 1000], "max_depth": [ 5, 8, 15, 25, 30, None],'max_features': ['auto', 'sqrt', 'log2', None, 0.8], 'criterion': ["entropy", "gini"]
     #}
    n_estimators = [500, 1000, 1500]
    max_features= [500, 1000, 1500]
    param_grid = dict(n_estimators=n_estimators, max_features=max_features)
     ## Define the n-folds for hyper-parameter optimization on training set.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=200889)
    #use out-of-bag samples ("oob_score= True") to estimate the generalization accuracy.
    model = RandomForestClassifier(bootstrap= True)
    #let's use cv=10 in the GridSearchCV call
    #performance estimation
    #initiate the grid
    grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv, scoring = 'roc_auc', n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    print('Best # of trees:', grid_result.best_estimator_.get_params()['n_estimators'])
    print('Best model:', grid_result.best_estimator_)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
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
    ## Plot mean ROC curve for cross-validation with n_splits=5 and n_repeats=100 to evaluate the variation of prediction in our training set.
    ## variable assignment to make it easier to read.
    X=x_train
    Y=y_train
    j=0
    for train, test in cv.split(X,Y):
        j=j+1
        print(j)
        probas_ = best_model.fit(X[train], Y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print("Train", roc_auc)

    ## Plot mean ROC curve for 100 epochs test set evaulation.
    probas_test = best_model.fit(x_train, y_train).predict_proba(x_test)
    # Compute ROC curve and area the curve
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, probas_test[:, 1])
    tprs_test.append(interp(mean_fpr_test, fpr_test, tpr_test))
    tprs_test[-1][0] = 0.0
    roc_auc_test = auc(fpr_test, tpr_test)
    aucs_test.append(roc_auc_test)
    print("Test", roc_auc_test)

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
plt.title('Random Forest ROC\n')
plt.legend(loc="lower right", fontsize=8)
#plt.show()
RF_plot.savefig('results/figures/Random_Forest_baxter.png', dpi=1000)
