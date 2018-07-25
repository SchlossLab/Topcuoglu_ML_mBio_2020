

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import *
import matplotlib.pyplot as plt
%matplotlib inline
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
y = np.expand_dims(y, axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=82089)


##scaler = StandardScaler()
##transformed = scaler.fit_transform(x_train)
##train = data_utils.TensorDataset(torch.from_numpy(transformed).float(),
##                                 torch.from_numpy(y_train).float())
##dataloader = data_utils.DataLoader(train, batch_size=233, shuffle=False)


x_train = x_train.values
train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())

dataloader = data_utils.DataLoader(train, batch_size=233, shuffle=False)

def create_model(layer_dims):
    model = torch.nn.Sequential()
    for idx, dim in enumerate(layer_dims):
        if (idx < len(layer_dims) - 1):
            module = torch.nn.Linear(dim, layer_dims[idx + 1])
            init.xavier_normal_(module.weight)
            model.add_module("linear" + str(idx), module)
        else:
            model.add_module("sig" + str(idx), torch.nn.Sigmoid())
        if (idx < len(layer_dims) - 2):
            model.add_module("relu" + str(idx), torch.nn.ReLU())
    return model

##In a similar manor to the train set, let's now scale and prepare a test set to let us know how our predictions are going.

#scaler = StandardScaler()
#transformed = scaler.fit_transform(x_test)
x_test = x_test.values
#y_test = y_test.values
test_set = torch.from_numpy(x_test).float()
test_valid = torch.from_numpy(y_test).float()
## Create model and hyper parameters
dim_in = x_train.shape[1]
dim_out = 1
layer_dims = [dim_in, 20, 10, dim_out]

model = create_model(layer_dims)
loss_fn =  torch.nn.BCEWithLogitsLoss()
##loss_fn =  torch.nn.MSELoss(size_average=False)
learning_rate = 0.0007
batch_size=233
n_epochs = 500
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## Now run model
history = { "loss": [], "accuracy": [], "loss_val": [], "accuracy_val": [] }
for epoch in range(n_epochs):
    loss = None
    for idx, (minibatch, target) in enumerate(dataloader):
        y_pred = model(Variable(minibatch))
        loss = loss_fn(y_pred, Variable(target.float()))
        prediction = [1 if x > 0.5 else 0 for x in y_pred.data.numpy()]
        prediction = np.asarray(prediction)
        prediction = np.expand_dims(prediction, axis=1)
        correct = (prediction == target.numpy()).sum()
        y_val_pred = model(Variable(test_set))
        loss_val = loss_fn(y_val_pred, Variable(test_valid.float()))
        prediction_val = [1 if x > 0.5 else 0 for x in y_val_pred.data.numpy()]
        prediction_val = np.asarray(prediction_val)
        prediction_val = np.expand_dims(prediction_val, axis=1)
        correct_val = (prediction_val == test_valid.numpy()).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    history["loss"].append(loss.data[0])
    history["accuracy"].append(100 * correct / len(prediction))
    history["loss_val"].append(loss_val.data[0])
    history["accuracy_val"].append(100 * correct_val / len(prediction_val))
    print("Loss, accuracy, val loss, val acc at epoch", epoch + 1,history["loss"][-1], history["loss_val"][-1], history["accuracy"][-1], history["accuracy_val"][-1])





plt.plot(history['accuracy'])
plt.plot(history['accuracy_val'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['loss_val'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



import readline
with open('pyhistory.txt', 'w') as f:
    for i in range(readline.get_current_history_length()):
        f.write(readline.get_history_item(i + 1) + "\n")
