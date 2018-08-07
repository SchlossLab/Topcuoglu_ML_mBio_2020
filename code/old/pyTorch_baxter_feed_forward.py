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
## generate the train dataset as tensor
scaler = StandardScaler()
transformed = scaler.fit_transform(x_train)
train = data_utils.TensorDataset(torch.from_numpy(transformed).float(),
                                 torch.from_numpy(y_train.as_matrix()).float())
## this makes a dataloader to put into model
dataloader = data_utils.DataLoader(train, batch_size=233, shuffle=False)
## make the test data tensor
scaler = StandardScaler()
transformed = scaler.fit_transform(x_test)
test_set = torch.from_numpy(transformed).float()
test_valid = torch.from_numpy(y_test.as_matrix()).float()
testloader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                         shuffle=True)


# Fully connected neural network with one hidden layer
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


## Create model and hyper parameters
input_size = x_train.shape[1]
hidden_size = 100
num_classes = 1
model = NeuralNet(input_size, hidden_size, num_classes)
learning_rate = 0.0007
n_epochs = 200
## define optimizer and loss function
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
## run the model
history = { "loss": [], "accuracy": [], "loss_val": [], "accuracy_val": [], "TP":[],"TN":[],"FP":[],"FN":[], "TPR":[], "TNR":[] }
for epoch in range(n_epochs):
    loss = None
    for idx, (minibatch, target) in enumerate(dataloader):
        y_pred = model(Variable(minibatch))
        loss = loss_fn(y_pred, Variable(target.float().view(-1,1)))
        prediction = [1 if x > 0.5 else 0 for x in y_pred.data.numpy()]
        correct = (prediction == target.numpy()).sum()
            # This can be uncommented for a per mini batch feedback
        #history["loss"].append(loss.data[0])
        #history["accuracy"].append(100 * correct / len(prediction))
        y_val_pred = model(Variable(test_set))
        loss_val = loss_fn(y_val_pred, Variable(test_valid.float().view(-1,1)))
        prediction_val = [1 if x > 0.5 else 0 for x in y_val_pred.data.numpy()]
        correct_val = (prediction_val == test_valid.numpy()).sum()
        # This can be uncommented for a per mini batch feedback
        #history["loss_val"].append(loss_val.data[0])
        #history["accuracy_val"].append(100 * correct_val / len(prediction_val))
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(prediction)):
            if target.numpy()[i]==prediction[i]==1:
                TP += 1
            if prediction[i]==1 and target.numpy()[i]==0:
                FP += 1
            if target.numpy()[i]==prediction[i]==0:
                TN += 1
            if prediction[i]==0 and target.numpy()[i]==1:
                FN += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    history["loss"].append(loss.data[0])
    history["accuracy"].append(100 * correct / len(prediction))
    history["loss_val"].append(loss_val.data[0])
    history["accuracy_val"].append(100 * correct_val / len(prediction_val))
    history["TP"].append(TP)
    history["TN"].append(TN)
    history["FP"].append(FP)
    history["FN"].append(FN)
    if (TP+FN) > 0:
        history["TPR"].append(TP/(TP+FN))
    if (FP+TN) > 0:
        history["TNR"].append(TN/(TN+FP))
    print("Loss, accuracy, val loss, val acc at epoch", epoch + 1,history["loss"][-1],
          history["accuracy"][-1], history["loss_val"][-1], history["accuracy_val"][-1] )



##Plot ROC
x = history["TPR"]
y = history["TNR"]
plt.plot(x, y)
plt.xlabel('True Negative Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
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
