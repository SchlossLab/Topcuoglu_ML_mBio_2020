## https://www.kaggle.com/kiranscaria/titanic-pytorch
## Add modules that are necessary
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier




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
#data= data[data.dx.str.contains("adenoma") == False]
data.rename(columns={'dx': 'class'}, inplace=True)
x = data.drop(["sample", "class", "numOtus", "label"], axis=1)
diagnosis = {"adenoma":1, "cancer":2, "normal":0}
y = data["class"].replace(diagnosis)
y.dropna()
x.dropna()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=True)
tpot = TPOTClassifier(generations=50, population_size=50, verbosity=2, cv=5, scoring='f1_macro')

tpot.fit(x_train, y_train)

print(tpot.score(x_test, y_test))
tpot.export('tpot_baxter_pipeline.py')
