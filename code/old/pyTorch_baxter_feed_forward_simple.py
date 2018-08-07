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
hidden_size = 100
num_classes = 2
learning_rate = 0.0001

import torch.optim as optim

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(epochs):
    inputs = Variable(torch.from_numpy(x_train.values)).float()
    labels = Variable(torch.from_numpy(y_train.values)).long()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print ("epoch #",epoch)
        print ("loss: ", loss.item())
        pred = torch.max(y_pred, 1)[1].eq(labels).sum()
        print ("acc:(%) ", 100*pred/len(inputs))
        loss.backward()
        optimizer.step()

train(50)

p_train = model(torch.from_numpy(x_train.values).float())
p_train = torch.max(p_train,1)[1]
len(p_train)
p_train = p_train.data.numpy()
accuracy_score(y_train, p_train)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def test(epochs):
    model.eval()
    input = Variable(torch.from_numpy(x_test.values)).float()
    label = Variable(torch.from_numpy(y_test.values)).long()
    for epoch in range(epochs):
        with torch.no_grad():
            y_preds = model(input)
            loss = criterion(y_preds, label)
            print ("epoch #",epoch)
            print ("loss: ", loss.item())
            preds = torch.max(y_preds, 1)[1].eq(label).sum()
            print ("acc (%): ", 100*preds/len(input))

test(50)

pred = model(torch.from_numpy(x_test.values).float())
pred = torch.max(pred,1)[1]
len(pred)
pred = pred.data.numpy()
accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
