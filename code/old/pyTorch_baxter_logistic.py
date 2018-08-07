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
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=82089)

x_train = x_train.as_matrix()
y_train = y_train.as_matrix()

## generate the train dataset as tensor
class PrepareData(Dataset):
    def __init__(self, x_train, y_train):
        if not torch.is_tensor(x_train):
            self.x_train = torch.from_numpy(x_train)
        if not torch.is_tensor(y_train):
            self.y_train = torch.from_numpy(y_train)
    def __len__(self):
        return len(self.x_train)
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

ds = PrepareData(x_train=x_train, y_train=y_train)
ds = DataLoader(ds, batch_size=100, shuffle=True)


class FeedForward(nn.Module):
    def __init__(self, n_features, n_neurons):
        super(FeedForward, self).__init__()
        self.hidden = nn.Linear(in_features=n_features, out_features=n_neurons)
        self.out_layer = nn.Linear(in_features=n_neurons, out_features=1)
    def forward(self, X):
        out = F.relu(self.hidden(X))
        out = F.sigmoid(self.out_layer(out))
        return out

model = FeedForward(n_features=6920, n_neurons=50)

class Logistic(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x):
        out = F.sigmoid(self.linear(x))
        return out

model2 = Logistic(input_size=6920, num_classes=1)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print('Accuracy on the training subset: {:.3f}'.format(log_reg.score(x_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg.score(x_test, y_test)))


log_reg100 = LogisticRegression(C=100)
log_reg100.fit(x_train, y_train)
print('Accuracy on the training subset: {:.3f}'.format(log_reg100.score(x_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg100.score(x_test, y_test)))

log_reg001 = LogisticRegression(C=0.01)
log_reg001.fit(x_train, y_train)
print('Accuracy on the training subset: {:.3f}'.format(log_reg001.score(x_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(log_reg001.score(x_test, y_test)))





##  hyper parameters
cost_func = torch.nn.MSELoss(size_average=False)
import torch.optim as optim
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
n_epochs = 100

history = { "loss": [], "accuracy": [], "loss_val": [], "accuracy_val": [], "TP":[],"TN":[],"FP":[],"FN":[] }
for epoch in range(n_epochs):
    loss = None
    for idx, (_x, _y) in enumerate(ds):
        _x = Variable(_x).float()
        _y = Variable(_y).float()
        #========forward pass=====================================
        yhat = model(_x).float()
        loss = cost_func(yhat, _y.view(-1,1))
        ##
        prediction = [1 if x > 0.5 else 0 for x in yhat.data.numpy()]
        correct = (prediction == _y.numpy()).sum()
        # This can be uncommented for a per mini batch feedback
        #history["loss_val"].append(loss_val.data[0])
        #history["accuracy_val"].append(100 * correct_val / len(prediction_val))
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(prediction)):
            if _y.numpy()[i]==prediction[i]==1:
                TP += 1
            if prediction[i]==1 and _y.numpy()[i]==0:
                FP += 1
            if _y.numpy()[i]==prediction[i]==0:
                TN += 1
            if prediction[i]==0 and _y.numpy()[i]==1:
                FN += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    history["loss"].append(loss.data[0])
    history["accuracy"].append(100 * correct / len(prediction))
    #history["loss_val"].append(loss_val.data[0])
    #history["accuracy_val"].append(100 * correct_val / len(prediction_val))
    history["TP"].append(TP)
    history["TN"].append(TN)
    history["FP"].append(FP)
    history["FN"].append(FN)
    print("Loss, accuracy, val loss, val acc at epoch", epoch + 1,history["loss"][-1], history["accuracy"][-1])


##Plot TP
plt.plot(history['TP'])
plt.plot(history['TN'])
plt.plot(history['FP'])
plt.plot(history['FN'])
plt.legend(['TP', 'TN','FP','FN'], loc='upper left')
plt.xlabel('epoch')
#plt.savefig('results/figures/accuracy_feed_forward.png')
plt.show()

##if you want to plot the accuracy
plt.plot(history['accuracy'])
plt.plot(history['accuracy_val'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('results/figures/accuracy_feed_forward.png')
plt.show()

##if you want to plot the loss
plt.plot(history['loss'])
plt.plot(history['loss_val'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('results/figures/loss_feed_forward.png')
plt.show()
