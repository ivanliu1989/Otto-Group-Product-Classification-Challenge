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
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import tanh
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
#from learning_rate import AdjustVariable
#from early_stopping import EarlyStopping, EarlyStoppingNoValidation

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
layers0 = [('input', InputLayer),
           
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]
           
net0 = NeuralNet(layers=layers0,                 
                 input_shape=(None, num_features),
                 
                 dense0_num_units=726,
                 dense0_nonlinearity=rectify,
                 dense0_W=lg.init.Uniform(),

                 dropout0_p=0.1,

                 dense1_num_units=243,
                 dense1_nonlinearity=rectify,
                 dense1_W=lg.init.Uniform(),

                 dropout1_p=0.2,
                 
                 dense2_num_units=81,
                 dense2_nonlinearity=rectify,
                 dense2_W=lg.init.Uniform(),
                 
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 output_W=lg.init.Uniform(),

                 #on_epoch_finished=[
                 #       AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                 #       AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 #       EarlyStopping(patience=200),
                 #       ],
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=50)
                 
net0.fit(X, y)
# 0.489205 200 0.5 150 0.5 100 0.01
# 0.480746 320 0.5 160 0.5 80 0.01
# 0.476469 726 0.5 243 0.5 81 0.01 (50)
# 0.472083 726 0.5 363 0.5 182 0.01 (47)

# Submission 
make_submission(net0, X_test, ids, encoder)