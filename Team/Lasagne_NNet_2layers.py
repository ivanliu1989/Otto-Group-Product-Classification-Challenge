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
from sklearn.metrics import log_loss

def Anscombe_Transform(data):
    data = (data + 0.375)**(0.5)
    return data
    
def load_train_data(path):
    df = pd.read_csv(path)
    df.ix[:,1:94].apply(Anscombe_Transform)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    return X, y, encoder
    
def load_fold0_data(path):
    df = pd.read_csv(path)
    df.ix[:,1:94].apply(Anscombe_Transform)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels, ids = X[:, 1:-1].astype(np.float32), X[:, -1],X[:, 0].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    return X, y, ids
    
def load_test_data(path):
    df = pd.read_csv(path)
    df.ix[:,1:94].apply(Anscombe_Transform)
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

# Train
for i in range(1,31):
    np.random.seed(9*i)
    X, y, encoder = load_train_data('../../train.csv')
    Test, ids = load_test_data('../../test.csv')
    
    X_train, y_train, encoder = load_train_data('../../train_fold0.csv')
    X_test, y_test, testIDS = load_fold0_data('../../test_fold0.csv')
    
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]
    
    for j in range(1,6):
        
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
                                AdjustVariable('update_momentum', start=0.85, stop=0.999),
                                EarlyStopping(patience=30)
                                ],
                         
                         eval_size=0.2,
                         verbose=1,
                         max_epochs=200)
        
                
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
        
                         update=nesterov_momentum,
                         update_learning_rate=theano.shared(float32(0.01)),
                         update_momentum=theano.shared(float32(0.9)),
                         
                         on_epoch_finished=[
                                AdjustVariable('update_learning_rate', start=0.015, stop=0.0001),
                                AdjustVariable('update_momentum', start=0.85, stop=0.999),
                                EarlyStopping(patience=30)
                                ],
                         
                         eval_size=0.1,
                         verbose=1,
                         max_epochs=200)
                         
        net0.fit(X_train, y_train)
        y_prob = net0.predict_proba(X_test)
        score=log_loss(y_test, y_prob)
        
        names = '../../Team_nnet/Val/valPred_Ivan_m'+str(i)+'_'+str(j)+'_CV'+ str(score)+'_nnet2.csv'
        submission = pd.DataFrame(data=y_prob, index=testIDS).sort_index(axis=1)
        submission.to_csv(names)
        print("Wrote submission to file {}.".format(names))
            
        # Submission 
        net1.fit(X, y)
        make_submission(net0, Test, ids, encoder, name='../../Team_nnet/Pred/testPred_Ivan_m'+str(i)+'_'+str(j)+'_nnet2.csv')
            