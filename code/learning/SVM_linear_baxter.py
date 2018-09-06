############## IMPORT MODULES ######################
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import *
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


parameters_svm = {'C':[0.9,0.01],'kernel':['rbf','linear'], 'gamma':[0,0.1,'auto'], 'probability':[True,False], 'random_state':[0,7,16],'decision_function_shape':['ovo','ovr'],'degree':[3,4,10]}

grid = GridSearchCV(estimator = model, param_grid = parameters, cv = 10, scoring = 'accuracy')
