# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import optimize

def arithmetic(tethas, X):
    return -1 * (np.log(X.dot(tethas)).mean())

def geometic(tethas, X):
    return -1 * (np.log((X ** tethas).prod(axis=1)).mean())

if __name__ == "__main__":
    trainset_path  = 'train_folds.csv'
    fold0_probs_dir = './probs/'
    train = pd.read_csv(trainset_path)
    labels = train.target.values
    folds = train.test_fold.values
    train = train.drop(['target', 'id', 'test_fold'], axis=1).values.astype(np.float32)

    lbl_enc = preprocessing.LabelEncoder()
    labelsPP = lbl_enc.fit_transform(labels)

    trainIDX = np.where(folds != 0)
    testIDX = np.where(folds == 0)

    X_train = train[trainIDX]
    X_test = train[testIDX]
    y_train = labelsPP[trainIDX]
    y_test = labelsPP[testIDX]

    columns = []
    loss = np.array([]).reshape((y_test.shape[0], 0))
    probs = np.array([]).reshape((y_test.shape[0], 0))
    probsC = np.array([]).reshape((y_test.shape[0], 0))
    for f in os.listdir(fold0_probs_dir):
        prob = pd.DataFrame.from_csv('./probs/'+f).values
        prob[prob < 1e-15] = 1e-15
        prob[prob > (1 - 1e-15)] = (1 - 1e-15)
        probs = np.c_[probs, prob]
        probsC = np.c_[probsC, prob[np.arange(prob.shape[0]), y_test]]
        loss = np.c_[loss, -1 * np.log(prob[np.arange(prob.shape[0]), y_test])]
        columns.append("_".join(f.split('_')[1:3]))

    loss = pd.DataFrame(loss, columns=columns)
    print loss.corr()
    print 'minloss: ', loss.min(axis=1).mean()
    print 'logloss: \n', loss.mean()
    x0 = np.ones(probsC.shape[1]) / probsC.shape[1]
    print 'AVG logloss: ', arithmetic(x0, probsC)


    bounds = [(0,1) for i in range(probsC.shape[1])]
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    unboundweigths = optimize.minimize(arithmetic,
                                       x0,
                                       args=(probsC),
                                       method='SLSQP',
                                       constraints=cons)

    boundweigths = optimize.minimize(arithmetic,
                                     x0,
                                     args=(probsC),
                                     method='SLSQP',
                                     bounds=bounds,
                                     constraints=cons)

    print 'unbounded weigths logloss: ', arithmetic(unboundweigths.x, probsC)

    uwpd = pd.Series(unboundweigths.x, index=columns)

    print 'models unbounded weigths: \n', uwpd

    print 'bounded weigths logloss: ', arithmetic(boundweigths.x, probsC)

    wpd = pd.Series(boundweigths.x, index=columns)

    print 'models bounded weigths: \n', wpd

    bgw = optimize.minimize(geometic,
                                     x0,
                                     args=(probsC),
                                     method='SLSQP',
                                     bounds=bounds,
                                     constraints=cons)

    print 'bounded geometric weigths logloss: ', geometic(bgw.x, probsC)

    wgpd = pd.Series(bgw.x, index=columns)

    print 'models bounded weigths: \n', wgpd