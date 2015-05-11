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

def load_train_data(path):
    df = pd.read_csv(path)
    df.ix[:,1:94] = df.ix[:,1:94].apply(np.log1p)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels, groups = X[:, 1:-3].astype(np.float32), X[:, -3], X[:, -1].astype(np.int32)
    encoder = LabelEncoder()
    groups = encoder.fit_transform(groups).astype(np.int32)
    y = encoder.fit_transform(labels).astype(np.int32)
    return X, y, encoder, groups
    
def load_test_data(path):
    df = pd.read_csv(path)
    df.ix[:,1:94] = df.ix[:,1:94].apply(np.log1p)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    return X, ids
    
def make_submission(clf, X_test, ids, encoder, name='lasagne_nnet.csv'):
    y_prob = clf.predict_proba(X_test)
    submission = pd.read_csv('../data/sampleSubmission.csv')
    submission.set_index('id', inplace=True)
    submission[:] = y_prob
    submission.to_csv(name)
    print("Wrote submission to file {}.".format(name))

# Load Data    
X, y, encoder, groups = load_train_data('../data/train_folds_cascade.csv')
X_test, ids = load_test_data('../../test.csv')
num_groups = 3
num_classes = len(encoder.classes_)
num_features = X.shape[1]

num_rows = X.shape[0]
Comb = np.append(X, X_test, axis=0)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
Comb = scaler.fit_transform(Comb)
X = Comb[:num_rows,:]
X_test = Comb[num_rows:,:]

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
                 
net0.fit(X, groups)

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

IDX = np.where(groups == 0)
X_group1 = X[IDX]
y_group1 = y[IDX]
IDX = np.where(groups == 1)
X_group2 = X[IDX]
y_group2 = y[IDX]

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
y_prob1 = net1.predict_proba(X_group1)

# Train Group 2
layers2 = [('input', InputLayer),
           ('dropoutf', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]
           
net2 = NeuralNet(layers=layers2,                 
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
                 
net2.fit(X_group2, y_group2)
y_prob2 = net2.predict(X_group2)


# Submission 
make_submission(net0, X_test, ids, encoder, name='../../Team_nnet_raw/lasagne_nnet2_0.466493.csv')
