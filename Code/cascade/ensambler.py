# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:49:50 2015

@author: rcrescenzi
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

if __name__ == "__main__":
    ############################ CONFIG #######################################
    train_path = '../train_folds.csv'
    G_path = './probs/valPred_Nickel_G_m1_CV0.0846_rf_150_cali_4.csv'
    G0_path = './probs/valPred_Nickel_G0_m1_CV0.5495_GB_150_LR0.05_MDnone_SS0.5_MFauto_MWFL0.001.csv'
    G1_path = './probs/valPred_Nickel_G1_m1_CV0.4831_GB_150_LR0.05_MDnone_SS0.5_MFauto_MWFL0.001.csv'
    target_path = '../probs/'
    user = 'Nickel'
    model_num = 2
    file_name = 'valPred_%s_cascade_m%i_CV%s.csv'%(user, model_num, '%1.4f')

    ##########################################################################

    train = pd.DataFrame.from_csv(train_path)
    labels = train.target.values
    folds = train.test_fold.values
    trainIDX = np.where(folds != 0)[0]
    testIDX = np.where(folds == 0)[0]
    ids = train[folds == 0].index.values

    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)

    y_test = labels[testIDX]

    probsG = pd.DataFrame.from_csv(G_path).values
    probsG0 = pd.DataFrame.from_csv(G0_path).values
    probsG1 = pd.DataFrame.from_csv(G1_path).values

    probs = np.c_[probsG[:, 0] * probsG1[:, 0],
                  probsG[:, 1] * probsG0[:, 0],
                  probsG[:, 1] * probsG0[:, 1],
                  probsG[:, 1] * probsG0[:, 2],
                  probsG[:, 2],
                  probsG[:, 0] * probsG1[:, 1],
                  probsG[:, 0] * probsG1[:, 2],
                  probsG[:, 0] * probsG1[:, 3],
                  probsG[:, 0] * probsG1[:, 4]]

    print confusion_matrix(y_test, probs.argmax(axis=1))
    print accuracy_score(y_test, probs.argmax(axis=1))
    loss = log_loss(y_test, probs)
    print loss

    probs = pd.DataFrame(probs, columns=lbl_enc.classes_ , index=ids)
    probs.index.name = 'id'
    print 'saving file: ', target_path + file_name%(loss)
#    probs.to_csv(probs_path + file_name%(loss))
