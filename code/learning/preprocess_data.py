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

## Read in the data
def process_data():
    ## shared has our input features(OTUs)
    shared = pd.read_table("data/baxter.0.03.subsample.shared")
    ## meta has the labels(diagnosis)
    meta = pd.read_table("data/metadata.tsv")

    ## Check and visualize the data
    meta.head()
    shared.head()

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
    y.dropna()
    x.dropna()

    ## Generate empty lists to fill with AUC values for test-set
    tprs_test = []
    aucs_test = []
    mean_fpr_test = np.linspace(0, 1, 100)
