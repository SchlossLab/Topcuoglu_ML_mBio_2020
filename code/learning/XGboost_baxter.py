import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import matplotlib
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, validation_curve
from sklearn import linear_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy import interp
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
# dependencies for plotting
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib as mpl
# dependencies for statistic analysis
from scipy import stats
#importing our parameter tuning dependencies
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (cross_val_score, GridSearchCV, StratifiedKFold, ShuffleSplit )
#importing our dependencies for Feature Selection
from sklearn.feature_selection import (SelectKBest, chi2, RFE, RFECV)
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from collections import defaultdict
# Importing our sklearn dependencies for the modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_curve, auc)
from sklearn.neural_network import MLPClassifier
from itertools import cycle
from scipy import interp
import warnings
warnings.filterwarnings('ignore')


seed = 2008
np.random.seed(seed)
##read in the data
shared = pd.read_table("data/baxter.0.03.subsample.shared")
shared.head()
meta = pd.read_table("data/metadata.tsv")
##check and visualize the data
meta.head()
shared.head()
## remove unnecessary columns from meta
meta = meta[['sample','dx']]
##rename the column name "Group" to match the "sample" in meta
shared = shared.rename(index=str, columns={"Group":"sample"})
##merge the 2 datasets on sample
data=pd.merge(meta,shared,on=['sample'])
##remove adenoma samples
data= data[data.dx.str.contains("adenoma") == False]
##drop all except OTU columns for x
x = data.drop(["sample", "dx", "numOtus", "label"], axis=1)
## Cancer =1 Normal =0
diagnosis = { "cancer":1, "normal":0}
##generate y which only has diagnosis as 0 and 1
y = data["dx"].replace(diagnosis)
##drop if NA elements
y.dropna()
x.dropna()
##split the data to generate training and test sets %80-20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=82089)
#X=x_train.values
#Y=y_train.values


#param = {'max_depth': 1, 'eta': 0.9, 'gamma':0,'silent': 1, 'min_child_weight':1, 'objective': 'binary:logistic'}
#param['nthread'] = 4
#param['eval_metric'] = 'auc'


#clf = xgb.XGBClassifier()
#cv = cross_validation.KFold(len(x_train), n_folds=5, shuffle=True, random_state=1)
#grid = {'learning_rate':[0.0001, 0.001, 0.01, 0.1], 'reg_lambda':[0, 0.001, 0.01, 0.10, 0.50, 1], 'max_depth':[1, 2], 'n_estimators':[150, 200, 250, 300, 500],}

#search = GridSearchCV(estimator=clf, param_grid=grid, scoring='accuracy', n_jobs=1, refit=True, cv=cv)
#search.fit(x_train, y_train)

#scores = cross_validation.cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=1, scoring='accuracy')

#print(search.best_params_)
#print(search.best_score_)

#{'learning_rate': 0.01, 'max_depth': 1, 'n_estimators': 250, 'reg_lambda': 0.1}

clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.001, max_delta_step=0,
       max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=0.1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.8)


cv = StratifiedKFold(n_splits=5, shuffle=True)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
tprs_test = []
aucs_test = []
mean_fpr_test = np.linspace(0, 1, 100)

XGboost_plot = plt.figure()
epochs = 100
for epoch in range(epochs):
    X=x_train.values
    Y=y_train.values
    X_test= x_test.values
    Y_test= y_test.values
    i = 0
    for train, test in cv.split(X,Y):
        probas_ = clf.fit(X[train], Y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Epoch %d, ROC fold %d (AUC = %0.2f)' % (epoch, i, roc_auc))
        #i += 1
    probas_ = clf.predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, probas_[:, 1])
    tprs_test.append(interp(mean_fpr_test, fpr_test, tpr_test))
    tprs_test[-1][0] = 0.0
    roc_auc_test = auc(fpr_test, tpr_test)
    aucs_test.append(roc_auc_test)


plt.plot([0, 1], [0, 1], linestyle='--', color='green', label='Luck', alpha=.8)
mean_tpr_test = np.mean(tprs_test, axis=0)
mean_tpr_test[-1] = 1.0
mean_auc_test = auc(mean_fpr_test, mean_tpr_test)
std_auc_test = np.std(aucs_test)
plt.plot(mean_fpr_test, mean_tpr_test, color='r', label=r'Mean test ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_test, std_auc_test), lw=2, alpha=.8)
std_tpr_test = np.std(tprs_test, axis=0)
tprs_upper_test = np.minimum(mean_tpr_test + std_tpr_test, 1)
tprs_lower_test = np.maximum(mean_tpr_test - std_tpr_test, 0)
plt.fill_between(mean_fpr_test, tprs_lower_test, tprs_upper_test, color='tomato', alpha=.2, label=r'$\pm$ 1 std. dev.')
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC\n')
plt.legend(loc="lower right", fontsize=8)
#plt.show()
XGboost_plot.savefig('results/figures/XGBoost_Baxter.png', dpi=1000)

plot_importance(clf, max_num_features=10)
pyplot.show()
######
#y_pred = clf.predict(x_test.values)
#print("Performance Accuracy on the Testing data:", #round(clf.score(x_test.values, y_test) *100))
#print("Number of correct classifiers:", round(accuracy_score(y_test, #y_pred, normalize=False)))
#print("Classification accuracy: ", round(accuracy_score(y_test, y_pred, #normalize=True) * 100))
# The classification Report# The cla
#target_names = ['Benign [Class 0]', 'Malignant[Class 1]']
#print(classification_report(y_test, y_pred, target_names=target_names))
