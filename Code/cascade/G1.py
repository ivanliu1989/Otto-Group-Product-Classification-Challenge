# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:49:50 2015

@author: rcrescenzi
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble
from sklearn.calibration import CalibratedClassifierCV as cali
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    ############################ CONFIG #######################################
    train_path = '../train_folds.csv'
    probs_path = './probs/'
    user = 'Nickel'
    model_num = 1
    model_params = 'GB_150_LR0.05_MDnone_SS0.5_MFauto_MWFL0.001'
    file_name = 'valPred_%s_G1_m%i_CV%s_%s.csv'%(user, model_num, '%1.4f', model_params)

    rf = ensemble.RandomForestClassifier(n_estimators=10,
                                         criterion='gini',
                                         max_depth=None,
                                         max_features='auto',
                                         min_weight_fraction_leaf=0.0,
                                         bootstrap=False,
                                         n_jobs=-2,
                                         verbose=1)

    learner  = cali(base_estimator=rf, method='isotonic', cv=4)

#    learner  = ensemble.GradientBoostingClassifier(n_estimators=250,
#                                                   learning_rate=0.05,
#                                                   max_depth=None,
#                                                   subsample=0.5,
#                                                   max_features='auto',
#                                                   min_weight_fraction_leaf=0.001,
#                                                   verbose=True,
#                                                   warm_start=True)

    ##########################################################################

    train = pd.DataFrame.from_csv(train_path)
    labels = train.target.values
    folds = train.test_fold.values
    trainIDX = np.where(folds != 0)[0]
    testIDX = np.where(folds == 0)[0]
    ids = train[folds == 0].index.values
    train = train.drop(['target', 'test_fold'], axis=1).values.astype(np.float32)

    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)

    X_train = train[trainIDX]
    X_test = train[testIDX]
    y_train = labels[trainIDX]
    y_test = labels[testIDX]

    G = np.array([0, 5, 6, 7, 8])

    train_filter = np.array([i in G for i in y_train])
    test_filter = np.array([i in G for i in y_test])

    X_train = X_train[train_filter]
    y_train = y_train[train_filter]

    y_test = y_test[test_filter]

    for i in range(len(G)):
        y_train[y_train == G[i]] = i
        y_test[y_test == G[i]] = i

    print 'fitting...'
    learner.fit(X_train, y_train)

    print 'predicting...'
    probs = learner.predict_proba(X_test)

    print confusion_matrix(y_test, probs[test_filter].argmax(axis=1))
    print accuracy_score(y_test, probs[test_filter].argmax(axis=1))
    loss = log_loss(y_test, probs[test_filter]) * y_test.shape[0] / probs.shape[0]
    print loss

    probs = pd.DataFrame(probs, columns=G , index=ids)
    probs.index.name = 'id'
    print 'saving file: ', probs_path + file_name%(loss)
#    probs.to_csv(probs_path + file_name%(loss))
