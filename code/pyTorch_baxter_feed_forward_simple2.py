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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=82089)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

scaler = StandardScaler()
transformed = scaler.fit_transform(x_test)
test_set = torch.from_numpy(transformed).float()
test_valid = torch.from_numpy(y_test.as_matrix()).float()



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Hyper-parameters
input_size = 6920
hidden_size = 5
num_classes = 2
learning_rate = 0.0007
batch_size = 50
batch_no = len(x_train) // batch_size
import torch.optim as optim

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

from sklearn.utils import shuffle
def train(epochs):
    for epoch in range(epochs):
        if epoch % 5 == 0:
            print('Epoch {}'.format(epoch+1))
        x_train2, y_train2 = shuffle(x_train, y_train)
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            inputs = Variable(torch.from_numpy(x_train2[start:end])).float()
            labels = Variable(torch.from_numpy(y_train2.values[start:end])).long()
            model.train()
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print ("epoch #",epoch)
            print ("loss: ", loss.item())
            pred = torch.max(y_pred, 1)[1].eq(labels).sum()
            print ("acc:(%) ", 100*pred/len(inputs))
            loss.backward()
            optimizer.step()

train(3)

p_train = model(torch.from_numpy(x_train).float())
p_train = torch.max(p_train,1)[1]
len(p_train)
p_train = p_train.data.numpy()
accuracy_score(y_train, p_train)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def test(epochs):
    model.eval()
    input = Variable(torch.from_numpy(x_test)).float()
    label = Variable(torch.from_numpy(y_test.values)).long()
    for epoch in range(epochs):
        with torch.no_grad():
            y_pred = model(input)
            loss = criterion(y_pred, label)
            print ("epoch #",epoch)
            print ("loss: ", loss.item())
            pred = torch.max(y_pred, 1)[1].eq(label).sum()
            print ("acc (%): ", 100*pred/len(input))


test(10)

pred = model(torch.from_numpy(x_test).float())
pred = torch.max(pred,1)[1]
len(pred)
pred = pred.data.numpy()
accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
