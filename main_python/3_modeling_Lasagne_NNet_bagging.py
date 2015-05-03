# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:52:55 2015

@author: ivanliu
"""

import pickle

np.random.seed(1719) # any random seed to differ bagging
net0.fit(X[:tr, ], y[:tr])
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