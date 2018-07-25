import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
%matplotlib inline

shared = pd.read_table("/Users/btopcuoglu/Documents/new_project/data/mothur/glne007.final.opti_mcc.unique_list.0.03.subsample.0.03.filter.shared")
shared.head()
meta = pd.read_table("/Users/btopcuoglu/Documents/new_project/data/mothur/metadata.tsv")
meta.head()
meta = meta[['sample','dx']]
shared = shared.rename(index=str, columns={"Group":"sample"})
data=pd.merge(meta,shared,on=['sample'])
data= data[data.dx.str.contains("adenoma") == False]
x = data.drop(["sample", "dx", "numOtus", "label"], axis=1)
diagnosis = { "cancer":1, "normal":0}
y = data["dx"].replace(diagnosis)
y.dropna()
x.dropna()
#y = np.expand_dims(y, axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=82089)

import os
from torch.utils.data import Dataset, DataLoader
class BaxterDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        X = self.X.iloc[idx].as_matrix().astype('double')
        X = X.reshape(-1, 417)
        Y = self.Y.iloc[idx].astype('double')
        return torch.from_numpy(X).double(), int(Y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(417, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 20)
        self.fc5 = nn.Linear(20, 10)
        self.fc6 = nn.Linear(10, 2)
    def forward(self, x):
        x = self.fc1(x).clamp(min=0)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


net = Net()

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


dataset = BaxterDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=10,
                        shuffle=True)
print(dataloader.dataset[0][0].shape)
print(dataloader.dataset[0][0].size())
print(type(dataloader.dataset[0][0]))
print(type(dataloader.dataset[0][1]))

dataset_test = BaxterDataset(x_test, y_test)
dataloader_test = DataLoader(dataset_test, batch_size=10,
                        shuffle=True)
print(dataloader.dataset[0][0].shape)
print(dataloader.dataset[0][0].size())
print(type(dataloader.dataset[0][0]))
print(type(dataloader.dataset[0][1]))



history = { "loss": [], "loss_val": []}
for epoch in range(500):  # loop over the dataset multiple times
    loss = None
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs.float()), Variable(labels.float())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        labels = labels.long()
        loss = criterion(outputs[:,0], labels)
        loss.backward()
        optimizer.step()
    history["loss"].append(loss.data[0])
    print("Loss at epoch", epoch + 1,history["loss"][-1])

for epoch in range(500):  # loop over the dataset multiple times
    loss = None
    for i, data in enumerate(dataloader_test, 0):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs.float()), Variable(labels.float())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        labels = labels.long()
        loss = criterion(outputs[:,0], labels)
        loss.backward()
        optimizer.step()
    history["loss_val"].append(loss.data[0])
    print("Loss at epoch", epoch + 1,history["loss_val"][-1])

plt.plot(history['loss'])
plt.plot(history['loss_val'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


##not working at the moment
correct = 0
total = 0
with torch.no_grad():
    for data in dataloader:
        inputs, labels = data
        outputs = net(inputs.float())
        _, predicted = torch.max(outputs.data.float(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network: %d %%' % (100 * correct / total))
