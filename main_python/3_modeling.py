# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:38:01 2015

@author: ivan
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC

np.random.seed(888)

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    from:
    https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/forums/t/2644/multi-class-log-loss-function

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota


def load_test_data(path=None, log=True, scale=True):
    if path is None:
        df = pd.read_csv('../../test.csv')
    else:
        df = pd.read_csv(path)
    
    if log:
        df.ix[:,1:94] = df.ix[:,1:94].apply(np.log1p)
        
    X = df.values
    
    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)


def load_train_data(path=None, train_size=0.7, log=True, scale=True, shuffle=True):
    if path is None:
       df = pd.read_csv('../../train.csv')
    else:
        df = pd.read_csv(path)
    
    if log:
        df.ix[:,1:94] = df.ix[:,1:94].apply(np.log1p)
        
    X = df.values.copy()
    
    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        
    if shuffle:
        np.random.shuffle(X)
        
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size,
    )
    print(" -- Loaded data.")
    return (X_train.astype(float), X_valid.astype(float),
            Y_train.astype(str), Y_valid.astype(str))


train_df, test_df, train_y, test_y = load_train_data()
encoder = LabelEncoder()
test_y = encoder.fit_transform(test_y)
train_y = encoder.fit_transform(train_y)

best = 10.    
for n in [450,500,600,700,800,900,1000]:
    #clf = RandomForestClassifier(n_estimators=n, max_depth=38, max_features=26, verbose=0) #
    #clf = PLSRegression(n_components=n, scale=True, tol=1e-06, max_iter=500)
    #clf= svm.LinearSVC(multi_class='crammer_singer',loss='l2',C=1,penalty='l2',tol=1e-4,dual=False,fit_intercept=True,intercept_scaling=1)
    clf= SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, verbose=True, max_iter=-1) #coef0=0.0, class_weight=None, random_state=None
    clf.fit(train_df,train_y)
    y_pred = clf.predict_proba(test_df)
    loss = multiclass_log_loss(test_y,y_pred)
    if loss < best:
        n_best = n
        best = loss
        postfix = '(*)'
    else:
        postfix = ''
    print ('comps: {:02d}\tLoss:{:5.4f} {}'.format(n,loss,postfix))


clf = RandomForestClassifier(n_estimators=n_best, max_depth=8, max_features=20, verbose=0)
clf.fit(train,targets)
y_pred = clf.predict_proba(test)


submission = pd.read_csv('sampleSubmission.csv')
submission.set_index('id', inplace=True)
submission[:] = y_pred
submission.to_csv('benchmark_pls.csv')
