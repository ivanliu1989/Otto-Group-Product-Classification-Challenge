# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:18:38 2015

@author: Ivan
"""
import numpy as np
import pandas as pd
import lasagne as lg
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

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
    
def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))

# Load Data    
X, y, encoder, scaler = load_train_data('../../train.csv')
X_test, ids = load_test_data('../../test.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

# Train
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           #('dense1', DenseLayer),
           ('output', DenseLayer)]
           
net0 = NeuralNet(layers=layers0,                 
                 input_shape=(None, num_features),

                 dense0_num_units=1000,
                 dense0_nonlinearity=rectify,
                 dense0_W=lg.init.Uniform(),

                 dropout_p=0.5,

                 #dense1_num_units=50,
                 
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 output_W=lg.init.Uniform(),

                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=100)
                 
net0.fit(X, y)

# Submission 
make_submission(net0, X_test, ids, encoder)
