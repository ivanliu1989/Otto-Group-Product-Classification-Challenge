# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:18:37 2015

@author: ivanliu
"""
from os import listdir
from os.path import isfile, join
import numpy as np

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

train = pd.read_csv(path)  

df.sum(1)