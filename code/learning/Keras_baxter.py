## Add modules that are necessary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout



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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=100, init='uniform', activation='relu', input_dim=6920))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))
# Adding the second hidden layer
classifier.add(Dense(output_dim=100, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))
# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size=50, nb_epoch=100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
