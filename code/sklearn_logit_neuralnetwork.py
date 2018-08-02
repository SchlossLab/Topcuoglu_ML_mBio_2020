from sklearn.datasets import load_breast_cancer
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
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold



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
#y = np.eye(2, dtype='uint8')[y]
##drop if NA elements
y.dropna()
x.dropna()

##split the data to generate training and test sets %80-20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=82089, shuffle=True)

## Generate the logistic regression model
logreg = LogisticRegression()
## With K-fold our training data is divided into 5 parts, the prediction model is generated for the 4 parts, and tested on the 5th part
kfold = KFold(n_splits=5, random_state=82089)
cv_results = cross_val_score(logreg, x_train, y_train, cv=kfold)
print (cv_results.mean()*100, "%")

## Define regularization parameter
## The lower the value of C, the higher we penalize the coeeficients of our logstic regression
param_grid = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
grid = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=kfold)
grid.fit(x_train,y_train)
print (grid.best_estimator_.C)
print (grid.best_score_*100, "%")

## Generate Neural Network model
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', random_state=1, activation='logistic', hidden_layer_sizes=(100,))
kfold = KFold(n_splits=5,random_state=82089)
cv_results = cross_val_score(clf, x_train, y_train, cv=kfold)

print (cv_results.mean()*100, "%")
## Find the regularization parameter
param_grid = {"alpha":10.0 ** -np.arange(-4, 7)}
grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold)
grid.fit(x_train,y_train)
print (grid.best_estimator_.alpha)
print (grid.best_score_*100, "%")

## Now that we know the optimal alpha ve C values. Let's check the results of accuracy.

## For Logistic regression
from sklearn.model_selection import cross_val_predict
logreg = linear_model.LogisticRegression(C=0.001)
#kfold = KFold(n_splits=5,random_state=82089)
#cv_results = cross_val_score(logreg, x_train, y_train, cv=kfold)
#print (cv_results.mean()*100, "%")

logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)*100), "%")
y_score_lr = logreg.predict_proba(x_test)
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_score_lr[:, 1])
##Plot AUC
area_under_the_curve = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate, lw=3, label="ROC Curve (area={0:.2f})".format(area_under_the_curve))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

## confusion matrix on test data
y_pred = logreg.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of logreg classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)*100), "%")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


## For Neural Network
clf = MLPClassifier(solver='lbfgs', random_state=1, activation='logistic', alpha=1000, hidden_layer_sizes=(100,))
#kfold = KFold(n_splits=5,random_state=82089)
#cv_results = cross_val_score(clf, x_train, y_train, cv=kfold)
#print (cv_results.mean()*100, "%")
## Fit the defined model to training data
clf.fit(x_train, y_train)
## predicted probabilities on test data
y_score_lr = clf.predict_proba(x_test)
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_score_lr[:, 1])
##Plot AUC
area_under_the_curve = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate, lw=3, label="ROC Curve (area={0:.2f})".format(area_under_the_curve))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

## confusion matrix on test data
y_pred = clf.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of neural net classifier on test set: {:.2f}'.format(clf.score(x_test, y_test)*100), "%")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# We can also see the misclassified examples of both models.
predicted = cross_val_predict(logreg, x_train, y_train, cv=kfold)
diff = predicted - y_train
misclass_indexes = diff[diff != 0].index.tolist()
print (misclass_indexes)

predicted = cross_val_predict(clf, x_train, y_train, cv=kfold)
diff = predicted - y_train
misclass_indexes = diff[diff != 0].index.tolist()
print (misclass_indexes)

############# Random Forest #######################
import numpy as np
import pandas as pd
# dependencies for plotting
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib as mpl
import seaborn as sns
# dependencies for statistic analysis
import statsmodels.api as sm
from scipy import stats
#importing our parameter tuning dependencies
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                    StratifiedKFold, ShuffleSplit )
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
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from itertools import cycle
from scipy import interp
import warnings
warnings.filterwarnings('ignore')
###### IMPORT MODULES #### ###
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

# Plot formatting

def plt_format():
    plt.rc('font', family='DejaVu Sans')
    plt.figure(figsize=(16,14))
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.size'] = 16
    sns.set(style="ticks", color_codes=True)
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['axes.linewidth'] = 1

plt_format()

#### CREATE MODEL ####
# Decide on the number of decision trees
param_grid = {
    'n_estimators': [ 25, 50, 100, 120, 150, 300, 500, 800, 1000], # the more parameters, the more computational expensive
     #"max_depth": [ 5, 8, 15, 25, 30, None],
    #'max_features': ['auto', 'sqrt', 'log2', None]
     }

#use out-of-bag samples ("oob_score= True") to estimate the generalization accuracy.
rfc = RandomForestClassifier(bootstrap= True, n_jobs= 1, oob_score= True)
#let's use cv=10 in the GridSearchCV call
#performance estimation
#initiate the grid
grid = GridSearchCV(rfc, param_grid = param_grid, cv=10, scoring ='accuracy')
#fit your data before you can get the best parameter combination.
grid.fit(x_train,y_train)
grid.cv_results_

# Let's find out the best scores, parameter and the estimator from the gridsearchCV
print("GridSearhCV best model:\n ")
print('The best score: ', grid.best_score_)
print('The best parameter:', grid.best_params_)
print('The best model estimator:', grid.best_estimator_)

#Let's find the optimal number of features and plot them.
plt_format()
# Setting the size of the plot
plt.figure(figsize=(14,8))

# model = RandomForestClassifier() with optimal values
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=True, random_state=None,
            verbose=0, warm_start=False)

# Create the RFE object and compute a cross-validated score
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(n_splits=3),
              scoring='accuracy', n_jobs=4)
rfecv.fit(x_train, y_train)

# Plot number of features VS. cross-validation scores
plt.title("Optimal Number of Features\n")
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_,  marker='o', linestyle='-', color='blue')

plt.show()
print("Optimal number of features : %d" % rfecv.n_features_)





















from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
mlp = MLPClassifier(random_state=42)
mlp.fit(x_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(mlp.score(x_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(mlp.score(x_test, y_test)))
