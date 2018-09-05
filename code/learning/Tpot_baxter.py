## https://www.kaggle.com/kiranscaria/titanic-pytorch
## Add modules that are necessary
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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import *
import matplotlib.pyplot as plt
import operator
from IPython.core.display import display
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
init_printing(use_unicode=True)
import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from torch.autograd import Variable
from scipy import interp
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_curve, auc)
from tpot import TPOTClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


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


tpot = TPOTClassifier(generations=50, population_size=50, verbosity=2, cv=5, n_jobs=1, scoring='roc_auc')

tpot.fit(x_train, y_train)

#Best pipeline: GradientBoostingClassifier(OneHotEncoder(MinMaxScaler(input_matrix), minimum_fraction=0.25, sparse=False), learning_rate=0.001, max_depth=10, max_features=0.45, min_samples_leaf=11, min_samples_split=2, n_estimators=100, subsample=0.8)

#TPOTClassifier(config_dict={'sklearn.ensemble.GradientBoostingClassifier': {'max_features': array([0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ]), 'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0], 'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7...  0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ])}, 'sklearn.preprocessing.RobustScaler': {}}, crossover_rate=0.1, cv=5, disable_update_check=False, early_stop=None, generations=10, max_eval_time_mins=5,max_time_mins=None, memory=None, mutation_rate=0.9, n_jobs=1,offspring_size=50, periodic_checkpoint_folder=None,population_size=50, random_state=None, scoring=None, subsample=1.0,verbosity=2, warm_start=False)

print(tpot.score(x_test, y_test))
tpot.predict(x_test)


y_pred = tpot.predict(x_test.values)
print("Performance Accuracy on the Testing data:", round(tpot.score(x_test.values, y_test) *100))
