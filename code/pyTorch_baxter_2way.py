%matplotlib inline

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

dataloader = data_utils.DataLoader(train, batch_size=233, shuffle=True)

# create a model class
# input->linear function->non linear function(sigmoid)->linear function->Softmax->CrossEntropy
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
test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
testloader = data_utils.DataLoader(test, batch_size=233, shuffle=True)

## Create model and hyper parameters
dim_in = x_train.shape[1]
dim_out = 1
layer_dims = [dim_in, 20, 10, dim_out]

model = create_model(layer_dims)
loss_fn =  torch.nn.BCEWithLogitsLoss()
#loss_fn =  torch.nn.MSELoss(size_average=False)
learning_rate = 0.0007
batch_size=233
n_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## Now run model
iter=0
correct_train = 0
total_train = 0
for epoch in range(n_epochs):
    for idx, (minibatch, target) in enumerate(dataloader):
        minibatch = Variable(minibatch)
        target = Variable(target.float())
        optimizer.zero_grad()
        outputs_train = model(minibatch)
        prediction = [1 if x > 0.5 else 0 for x in outputs_train.data.numpy()]
        prediction = np.asarray(prediction)
        prediction = np.expand_dims(prediction, axis=1)
        correct_train += (prediction == target.numpy()).sum()
        loss = loss_fn(outputs_train,target)
        loss.backward()
        optimizer.step()
        total_train += target.size(0)
        accuracy_train = 100 * correct_train / total_train
        iter+=1
        print('Iteration: {}. Loss: {}. Correct: {}. Accuracy: {}'.format(iter, loss.data[0], correct_train.data[0], accuracy_train))



if iter %500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for minibatch_test, target_test in testloader:
                minibatch = Variable(minibatch_test)
                # Forward pass only to get logits/output
                outputs = model(minibatch_test)
                # Get predictions from the maximum value
                prediction_val = [1 if x > 0.5 else 0 for x in outputs.data.numpy()]
                prediction_val = np.asarray(prediction_val)
                prediction_val = np.expand_dims(prediction_val, axis=1)
                correct += (prediction_val == target_test.numpy()).sum()
                # Total number of labels
                total += target_test.size(0)
                accuracy = 100 * correct / total
                    # Print Loss
            print('%Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))
