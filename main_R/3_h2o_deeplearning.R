setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
library(data.table);library(h2o);library(caret)
source('main_R/2_logloss_func.R')
load(file='data/target.RData');
train <- data.frame(fread('../train.csv', header=T, stringsAsFactor = F))
test <- data.frame(fread('../test.csv', header=T, stringsAsFactor = F))

localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '8g')
# h2o.clusterInfo(localH2O)
# h2o.rm(object= localH2O, keys= "DeepLearning_aa8d890913a2ad4d5f677dcad33849d2_xval2_holdout")
# h2o.ls(localH2O)
train <- data.frame(train)
trainIndex <- createDataPartition(train$target, p = .8,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]

train_df <- as.h2o(localH2O, train_df, key="train")
test_df <- as.h2o(localH2O, test_df, key="test")
test_raw <- as.h2o(localH2O, test, key="test_raw")

independent <- colnames(train_df[,2:(ncol(train_df)-1)])
dependent <- "target"
# fit <- h2o.gbm(y = dependent, x = independent, data = train_df, 
#                n.trees = 300, interaction.depth = 6,
#                shrinkage = 0.05, distribution= "multinomial")
# n.bins, balance.classes, n.minobsinnode = 2, 

# fit <- h2o.deeplearning(y = dependent, x = independent, data = train_df, 
#                         classification=T,activation="TanhWithDropout",#TanhWithDropout Rectifier
#                         input_dropout_ratio = 0.2,hidden_dropout_ratios = c(0,0,0),seed=8,
#                         hidden=c(100,90,20),epochs=1,variable_importances=F,rate_decay=0.66,rate=0.1,
#                         override_with_best_model=T,loss='CrossEntropy',nesterov_accelerated_gradient=T,
#                         l2=3e-6,shuffle_training_data=T,max_w2=4, epsilon = 1e-10, rho = 0.99)
# ,nfolds=10,, train_samples_per_iteration = -2,l1=1e-5, 

fit <- h2o.randomForest(y = dependent, x = independent, data = train_df, type = "BigData",
                        classification=T, ntree=1800, depth=50, mtries=10,
                        sample.rate=0.8, nbins = 30, seed=8,verbose=T)
# nodesize=10, validation=

pred <- h2o.predict(object = fit, newdata = test_df)
pred <- h2o.predict(object = fit, newdata = test_raw)
pred_ensemble = format(as.data.frame(pred[,2:10]), digits=2,scientific=F) # shrink the size of submission
target_df <- target[-trainIndex,]
MulLogLoss(target_df,data.matrix(pred_ensemble))

pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))
write.csv(pred_ensemble,file='../randomforest_0569.csv', quote=FALSE,row.names=FALSE)

h2o.shutdown(localH2O)
# 0.5186584 gbm
# 0.5547743 rf mtries=30
# 0.5400635 rf mtries=22, ntree=2000, depth=80
# 0.536 rf mtries=30, ntree=2000, depth=50

# 1.462512 deepLearning
# 0.9236796 hidden=c(512,512),rate_decay=0.3,input_dropout_ratio = 0.2,hidden_dropout_ratios = c(0.5,0.5),l1=3e-6,l2=6e-6,max_w2=4
# 0.8025674 input_dropout_ratio = 0.2,hidden_dropout_ratios = c(0.5,0.5,0.5),hidden=c(500,500,500),
# epochs=5,variable_importances=F,rate_decay=0.3,rate=0.1,nesterov_accelerated_gradient=T,
# l1=3e-6, l2=6e-6,shuffle_training_data=T,max_w2=4