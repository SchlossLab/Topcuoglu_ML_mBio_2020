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
from sklearn.utils import shuffle



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




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6920, 100)
        self.fc2 = nn.Linear(100, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x



net = Net()

batch_size = 50
num_epochs = 15
learning_rate = 0.001


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)


pyTorch_plot = plt.figure()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
tprs_test = []
aucs_test = []
mean_fpr_test = np.linspace(0, 1, 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test.as_matrix()).float()
train = data_utils.TensorDataset(torch.from_numpy(x_train).float(),
                                 torch.from_numpy(y_train.as_matrix()).float())
dataloader = data_utils.DataLoader(train, batch_size=50, shuffle=False)

for epoch in range(num_epochs):


    for idx, (minibatch, target) in enumerate(dataloader):
        ypred_var = net(Variable(minibatch))
        loss =criterion(ypred_var, Variable(target.long()))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct_num = 0
        ## The outputs of the model (ypred_var) are energies for the 10 classes. Higher the energy for a class, the more the network thinks that the image is of the particular class. So, let’s get the index of the highest energy:
        values, labels = torch.max(ypred_var, 1)
        correct_num = np.sum(labels.data.numpy() == target.numpy())
        fpr, tpr, thresholds = roc_curve(target.numpy(), labels.data.numpy())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('Epoch [%d], Loss:%.4f, Accuracy:%.4f' % (epoch, loss.data[0], correct_num/len(labels)))
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (epoch, roc_auc))

        with torch.no_grad():
            ypred_var_test = net(x_test)
            correct_num_test = 0
            values_test, labels_test = torch.max(ypred_var_test, 1)
            correct_num_test = np.sum(labels_test.data.numpy() == ypred_var_test.numpy())
            fpr_test, tpr_test, thresholds_test = roc_curve(y_test.numpy(), labels_test.data.numpy())
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
plt.title('PyTorch Neural Network ROC\n')
plt.legend(loc="lower right", fontsize=8)
#plt.show()
pyTorch_plot.savefig('results/figures/pyTorch_Baxter.png', dpi=1000)
