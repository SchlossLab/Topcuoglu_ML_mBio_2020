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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier





shared = pd.read_table("data/baxter.0.03.subsample.shared")
meta = pd.read_table("data/metadata.tsv")
## Remove unnecessary columns from meta and only keep label for classification(diagnosis) and the sample name
meta = meta[['sample','dx']]
## Rename the column name "Group" to match the "sample" in meta
shared = shared.rename(index=str, columns={"Group":"sample"})
## Merge the 2 datasets on sample
data=pd.merge(meta,shared,on=['sample'])
## Drop all except OTU columns for x
x = data.drop(["sample", "dx", "numOtus", "label"], axis=1)
## Cancer =1 Normal =0
diagnosis = {"adenoma":1, "cancer":2, "normal":0}
## Generate y which had diagnosis 0, 1, 2
y = data["dx"].replace(diagnosis)
# y = np.eye(2, dtype='uint8')[y]
## Drop if NA elements
y = y.dropna()
y= y.values
x = x.dropna()
x = x.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)
## Scale the dataset by removing mean and scaling to unit variance
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


c_values = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
param_grid = dict(C=c_values)
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=200889)
model = linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial")

## Based on the chosen model, create a grid to search for the optimal model
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv, scoring="accuracy", n_jobs=-1)
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
