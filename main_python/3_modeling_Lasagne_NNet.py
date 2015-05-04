# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:18:38 2015

@author: Ivan
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
from lasagne.nonlinearities import leaky_rectify#rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from adjust_variable import AdjustVariable
from early_stopping import EarlyStopping
from sklearn.decomposition import PCA

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

#pca = PCA()
#X = pca.fit_transform(X)
#X_test = pca.fit_transform(X_test)

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
                 #dense0_nonlinearity=leaky_rectify,
                 #dense0_W=lg.init.Uniform(),

                 dropout0_p=0.25,

                 dense1_num_units=500,
                 #dense1_nonlinearity=leaky_rectify,
                 #dense1_W=lg.init.Uniform(),

                 dropout1_p=0.25,
                 
                 dense2_num_units=300,
                 #dense2_nonlinearity=leaky_rectify,
                 #dense2_W=lg.init.Uniform(),
                 
                 dropout2_p=0.25,
                 
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 #output_W=lg.init.Uniform(),

                 update=nesterov_momentum,
                 #update=adagrad,
                 update_learning_rate=theano.shared(float32(0.01)),
                 update_momentum=theano.shared(float32(0.9)),
                 
                 on_epoch_finished=[
                        AdjustVariable('update_learning_rate', start=0.015, stop=0.0001),
                        AdjustVariable('update_momentum', start=0.9, stop=0.999),
                        EarlyStopping(patience=30)
                        ],
                 
                 eval_size=0.01,
                 verbose=1,
                 max_epochs=200)
                 
net0.fit(X, y)
# 0.489205 200 0.5 150 0.5 100 0.01
# 0.480746 320 0.5 160 0.5 80 0.01
# 0.476469 726 0.5 243 0.5 81 0.01 (50)
# 0.472083 726 0.5 363 0.5 182 0.01 (47)
# 0.467849 726 0.5 243 0.5 81 0.01 (42)(adjustvariable, earlystopping) 
# 0.467454 726 0.5 243 0.5 81 0.01 (37)(adjustvariable, earlystopping)

# 0.474071 0.15 1000 0.25 500 0.25 (46)
# 0.430886 0.15 1000 0.25 500 0.25 (44)
# 0.467751 0.15 1000 0.25 500 0.25 (28)

# Bagging
import pickle

prob = net0.predict_proba(X[te:])
neg_log_loss = 0
for row in range(0,prob.shape[0]):
neg_log_loss += - math.log(prob[row, y[te+row]]) 
neg_log_loss /= prob.shape[0] 
print "%f" % neg_log_loss

fs = 'objs.pickle_ada_1000_500_1'

with open(fs, 'w') as f:
pickle.dump(prob, f)

# then to bag

fs1 = 'objs.pickle_ada_1000_500_1'

fs2 = 'objs.pickle_ada_1000_500_2'

fs3 = 'objs.pickle_ada_1000_500_3'

with open(fs1) as f:

p1 = pickle.load(f)
with open(fs2) as f:
p2 = pickle.load(f)
with open(fs3) as f:
p3 = pickle.load(f)

pred =(p1+p2+p3)/3
neg_log_loss = 0
for row in range(0,pred.shape[0]):
neg_log_loss += - math.log(pred[row, y[te+row]]) 
neg_log_loss /= pred.shape[0] 
print "%f" % neg_log_loss

# Submission 
make_submission(net0, X_test, ids, encoder)