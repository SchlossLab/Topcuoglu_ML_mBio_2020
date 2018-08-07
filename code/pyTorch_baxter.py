## https://www.kaggle.com/kiranscaria/titanic-pytorch
## Add modules that are necessary
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6920, 100)
        self.fc2 = nn.Linear(100, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x



net = Net()

batch_size = 50
num_epochs = 50
learning_rate = 0.001
batch_no = len(x_train) // batch_size


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

from sklearn.utils import shuffle
from torch.autograd import Variable
from scipy import interp
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_curve, auc)


pyTorch_plot = plt.figure()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
for epoch in range(num_epochs):
    x_train, y_train = shuffle(x_train, y_train)
    # Mini batch learning
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(x_train.values[start:end]))
        y_var = Variable(torch.LongTensor(y_train.values[start:end]))
        # Forward + Backward + Optimize
        ypred_var = net(x_var)
        loss =criterion(ypred_var, y_var)
        correct_num = 0
        ## The outputs of the model (ypred_var) are energies for the 10 classes. Higher the energy for a class, the more the network thinks that the image is of the particular class. So, let’s get the index of the highest energy:
        values, labels = torch.max(ypred_var, 1)
        correct_num = np.sum(labels.data.numpy() == y_var.numpy())
        fpr, tpr, thresholds = roc_curve(y_var.numpy(), labels.data.numpy())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (epoch, roc_auc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [%d], Loss:%.4f, Accuracy:%.4f' % (epoch, loss.data[0], correct_num/len(labels)))

plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PyTorch Neural Network ROC\n')
plt.legend(loc="lower right", fontsize=8)
#plt.show()
pyTorch_plot.savefig('results/figures/pyTorch_Baxter.png', dpi=1000)

# Evaluate the model
net.eval()
pred = net(torch.from_numpy(x_test.values).float())
pred = torch.max(pred,1)[1]
len(pred)
pred = pred.data.numpy()
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
