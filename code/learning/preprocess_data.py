#!/usr/bin/python
#
# Author: Begum Topcuoglu
# Date: 2018-09-26
#
# This script prepares the Baxter Dataset before training the model
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
############## PRE-PROCESS DATA ######################

# shared dataset contains the OTUs and meta dataset contains the labels
#user defines their own datasets in the main.py using this function
def process_data(shared, meta):
    ## Remove unnecessary columns from meta and only keep label for classification(diagnosis) and the sample name
    meta = meta[['sample','dx']]
    ## Rename the column name "Group" to match the "sample" in meta
    shared = shared.rename(index=str, columns={"Group":"sample"})
    ## Merge the 2 datasets on sample
    data=pd.merge(meta,shared,on=['sample'])
    ## Remove adenoma samples. We will do a 2 classification model, just for cancer and normal colon samples.
    data= data[data.dx.str.contains("adenoma") == False]
    ## Drop all except OTU columns for x
    x = data.drop(["sample", "dx", "numOtus", "label"], axis=1)
    ## Cancer =1 Normal =0
    diagnosis = { "cancer":1, "normal":0}
    ## Generate y which only has diagnosis as 0 and 1
    y = data["dx"].replace(diagnosis)
    # y = np.eye(2, dtype='uint8')[y]
    ## Drop if NA elements
    y = y.dropna()
    x = x.dropna()
    return x, y