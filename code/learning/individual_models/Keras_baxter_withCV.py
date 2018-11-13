## Change backend to tensorflow by editing $HOME/.keras/keras.json
## "backend": "tensorflow"

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
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler



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

Keras_plot = plt.figure()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
tprs_test = []
aucs_test = []
mean_fpr_test = np.linspace(0, 1, 100)
num_epochs=100




for epoch in range(num_epochs):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    sc = StandardScaler()
    X = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    Y=y_train.values
    cv = StratifiedKFold(n_splits=5, random_state=200889)
    for train, test in cv.split(X,Y):
        # Initialising the ANN
        classifier = Sequential()
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(output_dim=100, init='uniform', activation='relu', input_dim=6920))
        # Adding dropout to prevent overfitting
        classifier.add(Dropout(p=0.5))
        # Adding the second hidden layer
        classifier.add(Dense(output_dim=100, init='uniform', activation='relu'))
        # Adding dropout to prevent overfitting
        classifier.add(Dropout(p=0.5))
        # Adding the output layer
        classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
        # Compiling the ANN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Fitting the ANN to the Training set
        classifier.fit(X[train], Y[train], epochs=20, validation_data=(x_test,y_test), batch_size=50, verbose=1)

        y_pred = classifier.predict(X[test], verbose=1).ravel()
        fpr, tpr, thresholds = roc_curve(Y[test], y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (epoch, roc_auc))

    classifier.fit(x_train, y_train, epochs=20, batch_size=50, verbose=1)
    y_pred_test = classifier.predict(x_test, verbose = 1).ravel()
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test)
    roc_auc_test = auc(fpr_test, tpr_test)
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
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean cross validation ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='dodgerblue', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Keras Neural Network ROC\n')
plt.legend(loc="lower right", fontsize=8)
#plt.show()
Keras_plot.savefig('results/figures/Keras_Baxter.png', dpi=1000)
