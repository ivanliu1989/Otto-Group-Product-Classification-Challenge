# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:48:36 2015

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
from lasagne.nonlinearities import rectify#rectify

def load_train_data(path):
    train = pd.read_csv(path)  
    train.ix[:,1:94] = train.ix[:,1:94].apply(np.log1p)
    train = train.values.copy()
    np.random.shuffle(train)
    ids, train, labels, folds = train[:, 0], train[:, 1:-2].astype(np.float32), train[:, -2], train[:, -1]
    trainIDX = np.where(folds != 0)
    testIDX = np.where(folds == 0)
    X_train = train[trainIDX]
    X_test = train[testIDX]
    ids = ids[testIDX]
    encoder = LabelEncoder()
    labelsPP = encoder.fit_transform(labels)
    y_train = labelsPP[trainIDX].astype(np.int32)
    y_test = labelsPP[testIDX].astype(np.int32)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, encoder, ids,scaler
    
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
X_train, y_train, X_test, y_test, encoder, testIDS,scaler = load_train_data('../data/train_folds.csv')
Test, ids = load_test_data('../../test.csv',scaler)
num_classes = len(encoder.classes_)
num_features = X_train.shape[1]

num_rows = X_train.shape[0]
num_rows_t = X_test.shape[0]
Comb = np.append(X_train, X_test, axis=0)
#Comb = np.append(Comb, Test, axis=0)
#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#Comb = scaler.fit_transform(Comb)
#X_train = Comb[:num_rows,:]
#X_test = Comb[num_rows:(num_rows_t+num_rows),:]
X = Comb[:(num_rows_t+num_rows),:]
y = np.append(y_train, y_test, axis=0)
#Test = Comb[(num_rows_t+num_rows):,:]

# Train
for i in range(1,31):
    # Load Data    
    X_train, y_train, X_test, y_test, encoder, testIDS,scaler = load_train_data('../data/train_folds.csv')
    Test, ids = load_test_data('../../test.csv',scaler)
    num_classes = len(encoder.classes_)
    num_features = X_train.shape[1]
    
    num_rows = X_train.shape[0]
    num_rows_t = X_test.shape[0]
    Comb = np.append(X_train, X_test, axis=0)
    #Comb = np.append(Comb, Test, axis=0)
    #scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    #Comb = scaler.fit_transform(Comb)
    #X_train = Comb[:num_rows,:]
    #X_test = Comb[num_rows:(num_rows_t+num_rows),:]
    X = Comb[:(num_rows_t+num_rows),:]
    y = np.append(y_train, y_test, axis=0)
    #Test = Comb[(num_rows_t+num_rows):,:]
    
    for j in range(1,6):
        
        np.random.seed(8*i*j)
        
        layers0 = [('input', InputLayer),
                   ('dropoutf', DropoutLayer),
                   ('dense0', DenseLayer),
                   ('dropout0', DropoutLayer),
                   ('dense1', DenseLayer),
                   ('dropout1', DropoutLayer),
                   #('dense2', DenseLayer),
                   #('dropout2', DropoutLayer),
                   ('output', DenseLayer)]
                   
        net0 = NeuralNet(layers=layers0,                 
                         input_shape=(None, num_features),
                         dropoutf_p=0.15,
                         dense0_num_units=1000,
                         dense0_nonlinearity=rectify,
                         dropout0_p=0.25,
                         dense1_num_units=500,
                         dense1_nonlinearity=rectify,
                         dropout1_p=0.25,
                        # dense2_num_units=300,
                         #dense2_nonlinearity=rectify,
                         #dropout2_p=0.25,
                         output_num_units=num_classes,
                         output_nonlinearity=softmax,
                         output_W=lg.init.Uniform(),
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
            
        layers1 = [('input', InputLayer),
                   ('dropoutf', DropoutLayer),
                   ('dense0', DenseLayer),
                   ('dropout0', DropoutLayer),
                   ('dense1', DenseLayer),
                   ('dropout1', DropoutLayer),
                   #('dense2', DenseLayer),
                   #('dropout2', DropoutLayer),
                   ('output', DenseLayer)]
                   
        net1 = NeuralNet(layers=layers0,                 
                         input_shape=(None, num_features),
                         dropoutf_p=0.15,
                         dense0_num_units=1000,
                         dense0_nonlinearity=rectify,
                         dropout0_p=0.25,
                         dense1_num_units=500,
                         dense1_nonlinearity=rectify,
                         dropout1_p=0.25,
                        # dense2_num_units=300,
                         #dense2_nonlinearity=rectify,
                         #dropout2_p=0.25,
                         output_num_units=num_classes,
                         output_nonlinearity=softmax,
                         output_W=lg.init.Uniform(),
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
        
        # 0.467751 0.15 1000 0.25 500 0.25 (28)
        # 0.423485 0.15 800 0.25 500 0.25 300 0.25 (65)
        
        # 0.15 800 0.25 500 0.25 300 0.25 | 0.015-0.001 | 0.9-0.999 | 0.2 | 150 | log
        