# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:35:50 2015

@author: ivanliu
"""
import theano
import numpy as np
import pandas as pd
import lasagne as lg
from util import float32
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from adjust_variable import AdjustVariable
from early_stopping import EarlyStopping
from sklearn.metrics import log_loss

def load_train_data(path):
    train = pd.read_csv(path)  
    train.ix[:,1:94] = train.ix[:,1:94].apply(np.log1p)
    train = train.values.copy()
    np.random.shuffle(train)
    train, labels, folds, groups, ids = train[:, 1:-3].astype(np.float32), train[:, -3], train[:, -2], train[:, -1].astype(np.int32), train[:, 0].astype(int)
    trainIDX = np.where(folds != 0)
    testIDX = np.where(folds == 0)
    X_train = train[trainIDX]
    X_test = train[testIDX]
    ids = ids[testIDX]
    encoder = LabelEncoder()
    groups = encoder.fit_transform(groups).astype(np.int32)
    labelsPP = encoder.fit_transform(labels)
    y_train = labelsPP[trainIDX].astype(np.int32)
    y_test = labelsPP[testIDX].astype(np.int32)
    groups_train = groups[trainIDX]
    groups_test = groups[testIDX]
    
    return X_train, y_train, X_test, y_test, groups_train,groups_test,encoder,ids
    
# Load Data    
X, y, X_test, y_test, groups_train,groups_test,encoder,ids = load_train_data('../data/train_folds_cascade.csv')
num_groups = 3
num_classes = len(encoder.classes_)
num_features = X.shape[1]

#num_rows = X.shape[0]
#Comb = np.append(X, X_test, axis=0)
#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#Comb = scaler.fit_transform(Comb)
#X = Comb[:num_rows,:]
#X_test = Comb[num_rows:,:]

# Train Groups
np.random.seed(9)

layers0 = [('input', InputLayer),
           ('dropoutf', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]
           
net0 = NeuralNet(layers=layers0,                 
                 input_shape=(None, num_features),
                 dropoutf_p=0.15,
                 dense0_num_units=800,
                 dropout0_p=0.25,
                 dense1_num_units=500,
                 dropout1_p=0.25,                 
                 dense2_num_units=300,                 
                 dropout2_p=0.25,                 
                 output_num_units=3,
                 output_nonlinearity=softmax,
                 output_W=lg.init.Uniform(),
                 update=nesterov_momentum,
                 update_learning_rate=theano.shared(float32(0.01)),
                 update_momentum=theano.shared(float32(0.9)),
                 
                 on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=0.015, stop=0.001),
                        AdjustVariable('update_momentum', start=0.9, stop=0.999),
                        EarlyStopping(patience=30)
                        ],
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=200)
                 
net0.fit(X, groups_train)

y_prob = net0.predict(X_test)
IDX = np.where(y_prob == 0)
test_group1 = X_test[IDX]
ids1 = ids[IDX]
IDX = np.where(y_prob == 1)
test_group2 = X_test[IDX]
ids2 = ids[IDX]
IDX = np.where(y_prob == 2)
test_group3 = X_test[IDX]
ids3 = ids[IDX]

IDX = np.where(groups_train == 0)
X_group1 = X[IDX]
y_group1 = y[IDX]
IDX = np.where(groups_train == 1)
X_group2 = X[IDX]
y_group2 = y[IDX]
IDX = np.where(groups_train == 2)
X_group3 = X[IDX]
y_group3 = y[IDX]

y_prob_p = net0.predict_proba(X_test)
log_loss(groups_test, y_prob_p) #0.081823 | 0.0776

# Train Group 1
layers1 = [('input', InputLayer),
           ('dropoutf', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]
           
net1 = NeuralNet(layers=layers1,                 
                 input_shape=(None, num_features),
                 dropoutf_p=0.15,
                 dense0_num_units=800,
                 dropout0_p=0.25,
                 dense1_num_units=500,
                 dropout1_p=0.25,                 
                 dense2_num_units=300,                 
                 dropout2_p=0.25,                 
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 output_W=lg.init.Uniform(),
                 update=nesterov_momentum,
                 update_learning_rate=theano.shared(float32(0.01)),
                 update_momentum=theano.shared(float32(0.9)),
                 
                 on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=0.015, stop=0.001),
                        AdjustVariable('update_momentum', start=0.9, stop=0.999),
                        EarlyStopping(patience=30)
                        ],
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=200)
                 
net1.fit(X_group1, y_group1)
y_prob1 = net1.predict_proba(test_group1)

# Train Group 2
layers2 = [('input', InputLayer),
           ('dropoutf', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
            ('dense3', DenseLayer),
           ('dropout3', DropoutLayer),
           ('output', DenseLayer)]
           
net2 = NeuralNet(layers=layers2,                 
                 input_shape=(None, num_features),
                 dropoutf_p=0.15,
                 dense0_num_units=1000,
                 dropout0_p=0.25,
                 dense1_num_units=500,
                 dropout1_p=0.25,                 
                 dense2_num_units=500,                 
                 dropout2_p=0.25, 
                 dense3_num_units=300,                 
                 dropout3_p=0.25, 
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 output_W=lg.init.Uniform(),
                 update=nesterov_momentum,
                 update_learning_rate=theano.shared(float32(0.01)),
                 update_momentum=theano.shared(float32(0.9)),
                 
                 on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
                        AdjustVariable('update_momentum', start=0.9, stop=0.999),
                        EarlyStopping(patience=100)
                        ],
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=500)
                 
net2.fit(X_group2, y_group2)
y_prob2 = net2.predict_proba(test_group2)

# Train Group 3
layers3 = [('input', InputLayer),
           ('dropoutf', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]
           
net3 = NeuralNet(layers=layers3,                 
                 input_shape=(None, num_features),
                 dropoutf_p=0.15,
                 dense0_num_units=800,
                 dropout0_p=0.25,
                 dense1_num_units=500,
                 dropout1_p=0.25,                 
                 dense2_num_units=300,                 
                 dropout2_p=0.25,                 
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 output_W=lg.init.Uniform(),
                 update=nesterov_momentum,
                 update_learning_rate=theano.shared(float32(0.01)),
                 update_momentum=theano.shared(float32(0.9)),
                 
                 on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=0.015, stop=0.001),
                        AdjustVariable('update_momentum', start=0.9, stop=0.999),
                        EarlyStopping(patience=20)
                        ],
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=200)
                 
net3.fit(X_group3, y_group3)
y_prob3 = net3.predict_proba(test_group3)

# Submission 
y_prob_tot = net3.predict_proba(X_test)
IDX = np.where(y_prob == 0)
y_prob_tot[IDX] = y_prob1
y_test1 = y_test[IDX]
IDX = np.where(y_prob == 1)
y_prob_tot[IDX] = y_prob2
y_test2 = y_test[IDX]
IDX = np.where(y_prob == 2)
y_prob_tot[IDX] = y_prob3
y_test3 = y_test[IDX]

log_loss(y_test1, y_prob1) #0.43278267824509881
log_loss(y_test2, y_prob2) #0.90879075153076982
log_loss(y_test3, y_prob3)
log_loss(y_test, y_prob_tot)

