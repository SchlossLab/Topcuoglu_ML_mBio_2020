import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
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


# Make sure that the class is labeled 'target' in the data file
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
data.rename(columns={'dx': 'class'}, inplace=True)
x = data.drop(["sample", "class", "numOtus", "label"], axis=1)
diagnosis = { "cancer":1, "normal":0}
y = data["class"].replace(diagnosis)
y.dropna()
x.dropna()

# Score on the training set was:0.8492612704601008
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.8, min_samples_leaf=2, min_samples_split=2, n_estimators=100)),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.6000000000000001, min_samples_leaf=8, min_samples_split=7, n_estimators=100)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=13, min_samples_split=10)),
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.6500000000000001, n_estimators=100), step=0.7000000000000001),
    StackingEstimator(estimator=MultinomialNB(alpha=1.0, fit_prior=False)),
    ZeroCount(),
    StackingEstimator(estimator=LogisticRegression(C=0.1, dual=True, penalty="l2")),
    StackingEstimator(estimator=LogisticRegression(C=10.0, dual=True, penalty="l2")),
    BernoulliNB(alpha=0.01, fit_prior=True)
)

tprs_test = []
aucs_test = []
mean_fpr_test = np.linspace(0, 1, 100)

epochs= 100
for epoch in range(epochs):
    ## Split dataset to 80% training 20% test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=200889)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

## Converting to numpy array from pandas
    X=x_train.values
    Y=y_train.values
    X_test= x_test.values
    Y_test= y_test.values
    Tpot_plot = plt.figure()
## Plot mean ROC curve for cross-validation with n_splits=5 and n_repeats=100 to evaluate the variation of prediction in our training set.
    for train, test in cv.split(X,Y):
        probas_ = exported_pipeline.fit(X[train], Y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    ## Plot mean ROC curve for test set after each fitting of subset training set

## Plot mean ROC curve for never before seen test set.
    probas_ = exported_pipeline.predict_proba(x_test)
# Compute ROC curve and area the curve
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, probas_[:, 1])
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
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean cross-validation ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Tpot Defined Random Forest ROC\n')
plt.legend(loc="lower right", fontsize=8)
Tpot_plot.savefig('results/figures/Tpot_pipeline_Baxter.png', dpi=1000)

filename = 'finalized_tpot_model.sav'
joblib.dump(exported_pipeline, filename)
