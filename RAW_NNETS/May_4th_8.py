# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:18:38 2015
@author: Ivan
"""
import theano
import numpy as np
import pandas as pd
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
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler
    
def load_test_data(path, scaler):
    df = pd.read_csv(path)
    df.ix[:,1:94] = df.ix[:,1:94].apply(np.log1p)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids
    
def make_submission(clf, X_test, ids, encoder, name='lasagne_nnet.csv'):
    y_prob = clf.predict_proba(X_test)
    submission = pd.read_csv('../data/sampleSubmission.csv')
    submission.set_index('id', inplace=True)
    submission[:] = y_prob
    submission.to_csv(name)
    print("Wrote submission to file {}.".format(name))

# Load Data    
X, y, encoder, scaler = load_train_data('../../train.csv')
X_test, ids = load_test_data('../../test.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

# Train
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
                 
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,

                 update=nesterov_momentum,
                 update_learning_rate=theano.shared(float32(0.01)),
                 update_momentum=theano.shared(float32(0.9)),
                 
                 on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=0.015, stop=0.0001),
                        AdjustVariable('update_momentum', start=0.9, stop=0.999),
                        EarlyStopping(patience=20)
                        ],
                 
                 eval_size=0.1,
                 verbose=1,
                 max_epochs=100)
                 
net0.fit(X, y)

# Submission 
make_submission(net0, X_test, ids, encoder, name='../../lasagne_nnet_rect_0.474885.csv')
# 0.457593 71 39 58