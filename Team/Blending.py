# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:18:37 2015

@author: ivanliu
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

mypath='../../nnet_blending/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
onlyfiles = onlyfiles[1:]

for i in range(0,len(onlyfiles)):
    if i == 0:
        train = pd.read_csv(mypath+onlyfiles[i])  
    else:        
        train = train + pd.read_csv(mypath+onlyfiles[i])

train = train/len(onlyfiles)
train.ix[:,0] = train.ix[:,0].astype(np.int)
#train.ix[:,0] = train.ix[:,0]/len(onlyfiles)
train.to_csv('../../blending_'+str(len(onlyfiles))+'sum.csv', index=False)

train = train.values.copy()
train = train[:,1:].astype(np.float32)

log_loss(y_test,train)