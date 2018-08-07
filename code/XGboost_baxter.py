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

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)


param = {'max_depth': 1, 'eta': 0.9, 'gamma':0,'silent': 1, 'min_child_weight':1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

n_folds = 10
early_stopping = 10

cv = xgb.cv(param, dtrain, 5000, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)

evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 5
bst = xgb.train(param, dtrain, num_round, evallist)
ypred = bst.predict(dtest)

cv=StratifiedKFold(y_train, n_folds=10, shuffle=True, random_state=seed)
params_grid = {
    'max_depth':[1, 2, 3],
    'n_estimators':[5, 10, 25, 50],
    'learning_rate': np.linspace(0.00001, 1, 3)
}

params_fixed = {
    'objective':'binary:logistic',
    'silent':1
}

bst_grid = GridSearchCV(
    estimator=XGBClassifier(**params_fixed, seed=seed),
    param_grid=params_grid,
    cv=cv,
    scoring='accuracy'
)

bst_grid.fit(x_train,y_train)
print(bst_grid.best_estimator_.max_depth)
print(bst_grid.best_estimator_.n_estimators)
print(bst_grid.best_estimator_.subsample)
print(bst_grid.best_estimator_.colsample_bytree)
print(bst_grid.best_estimator_.learning_rate)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
classifier1=XGBClassifier(max_depth=1, n_estimators=10, subsample=1,
                         colsample_bytree=1, learning_rate=0.5)
classifier1.fit(x_train,y_train)
y_pred1=classifier1.predict(x_test)
from sklearn.metrics import confusion_matrix
print(accuracy_score(y_test, y_pred1))
cm=confusion_matrix(y_test,y_pred1)



print("Best accuracy obtained: {O}".format(bst_grid.best_score))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}:{}".format(key,value))


classifier=XGBClassifier()
classifier.fit(x_train,y_train)
