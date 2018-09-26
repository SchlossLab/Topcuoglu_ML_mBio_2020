#!/usr/bin/python
#
# Author: Begum Topcuoglu
# Date: 2018-09-26
#
# This script loads the Python modules necessary for Machine Learning project.
#

############## IMPORT MODULES ######################
def import_modules():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn import linear_model
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import xgboost
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import sympy 
    from scipy import interp
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.externals import joblib
    from sklearn.preprocessing import StandardScaler
