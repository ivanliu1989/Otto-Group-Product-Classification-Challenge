# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:18:37 2015

@author: ivanliu
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

mypath='../../Team_nnet/Val/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
onlyfiles = onlyfiles[1:]

for i in range(0,len(onlyfiles)):
    if i == 0:
        train = pd.read_csv(mypath+onlyfiles[i])  
    else:        
        train = train + pd.read_csv(mypath+onlyfiles[1])

train = train/len(onlyfiles[1:])
train.to_csv('xgb_blending_18.csv')

train = train.values.copy()
train = train[:,1:].astype(np.float32)

log_loss(y_test,train)