#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import scipy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as skl
import io
import pickle
import json
import seaborn as sns

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

from sklearn. preprocessing import scale


# In[2]:


def balance_data(data): 
    dfH = data.loc[data['BCandidate']==0,:]
    print("HC length: ",len(dfH))

    dfT =data.loc[data['BCandidate']==1,:]
    print("DT length: ",len(dfT))
    df_new = data
    A = len(dfT)
    B = len(dfH)
    
    if B>A: 
        print('look at function "balance_data"')
    missing_rows = A-B
    row_fraction = missing_rows/B
    print('missing_rows:', missing_rows, 'row_fraction:', row_fraction) 
    for i in range(0,math.floor(row_fraction),1):
        print('data repeat',i)
        df_new = df_new.append(dfH)
        
    leftover = row_fraction - math.floor(row_fraction)
    df_new = df_new.append(dfH[:math.floor(len(dfH)*leftover)])

    print("DT length ratio: ",len(dfT)/len(df_new))
    
    data = df_new
    
    return data


# In[3]:

def choose_model(name):
    granularity = 30
    if name =="support vector machine" or name == 'SVM':
        model = SVC(gamma='auto') 
        param_grid = { 
            'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            'C' : list(range(1,30, granularity))}
        

    elif name == 'logistic regression':## you can add parameter classed weight = balanced
        model = LogisticRegression(max_iter = 1000)
        param_grid={ 
            'class_weight' : ['none','balanced'],
            'penalty' : ['none'],#'elasticnet',
            'tol' : [1e-1,1e-6],
            'C' : list(range(1,30, granularity))}
        
    elif name == 'neural net' or name == 'NN':
        model = MLPClassifier(max_iter=1000)
        param_grid={ 
            'alpha' : [1e-3,1e-2,.1,1,10],## regularization
            'activation': ['tanh'],
            'learning_rate': ['adaptive']}
        
    else: 
        print("wrong name input")
        print("change granularity?")
    return model, param_grid



def model_training (model, X_train,Y_train,param_grid): 
    CVM = GridSearchCV(
        model, param_grid = param_grid,n_jobs = -1)

    CVM.fit(X_train,Y_train)
    return CVM


# In[4]:


def save_model(model,name, data_name):
    filename = name+'_GS_'+data_name
    #pickle.dump(model, open("trained_models/"+filename, 'wb'))
    pickle.dump(model, open(filename, 'wb'))
    return filename


# In[5]:


def scorer(filename, X_test, Y_test ): 
    loaded_model = pickle.load(open(filename, 'rb'))
    print(filename+" results on test data")
    #print("roc_auc:",roc_auc_score(Y_test,loaded_model.predict(X_test)))
    print("accurarcy:",accuracy_score(Y_test,loaded_model.predict(X_test)))
    #print("note 1 positive for Heart Disease")
    cm = confusion_matrix(Y_test,loaded_model.predict(X_test))
    ConfusionMatrixDisplay(cm).plot()
    
    json_ptr = open("scores/scores.json", 'r')
    scores = json.load(json_ptr)
    scores["roc_auc_"+filename] = roc_auc_score(Y_test,loaded_model.predict(X_test))
    scores["accurarcy_"+filename] = accuracy_score(Y_test,loaded_model.predict(X_test))
    
    json_ptr = open("scores/scores.json", 'w')
    json.dump(scores, json_ptr)
    json_ptr.close()
    return loaded_model


# In[7]:








