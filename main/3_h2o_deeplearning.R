setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
library(h2o);library(caret)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_log.RData') # raw_data_log_scale.RData

localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '8g')
# h2o.clusterInfo(localH2O)
# h2o.rm(object= localH2O, keys= "DeepLearning_aa8d890913a2ad4d5f677dcad33849d2_xval2_holdout")
# h2o.ls(localH2O)

trainIndex <- createDataPartition(train$target, p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]

train_df <- as.h2o(localH2O, train_df, key="train")
test_df <- as.h2o(localH2O, test_df, key="test")

independent <- colnames(train_df[,2:94])
dependent <- "target"
# fit <- h2o.gbm(y = dependent, x = independent, data = train_df, 
#                n.trees = 300, interaction.depth = 6,
#                shrinkage = 0.05, distribution= "multinomial")
# n.bins, balance.classes, n.minobsinnode = 2, 

fit <- h2o.deeplearning(y = dependent, x = independent, data = train_df, 
                        classification=T,activation="TanhWithDropout",#TanhWithDropout Rectifier
                        input_dropout_ratio = 0.2,hidden_dropout_ratios = c(0.5,0.5),
                        hidden=c(512,512),epochs=1,variable_importances=F,rate_decay=0.3,rate=0.1,
                        override_with_best_model=F,loss='CrossEntropy',nesterov_accelerated_gradient=F,
                        l1=3e-6, l2=6e-6,shuffle_training_data=T,max_w2=4)
# ,epsilon=0.1,nesterov_accelerated_gradient=F,nfolds=10,seed=8,adaptive_rate=0.9,

# fit <- h2o.randomForest(y = dependent, x = independent, data = train_df, 
#                         classification=T, ntree=500, depth=30, mtries=30,
#                         sample.rate=0.8, nbins=T, seed=8,verbose=T)
# nodesize=10,

pred <- h2o.predict(object = fit, newdata = test_df)
pred_ensemble = format(as.data.frame(pred[,2:10]), digits=2,scientific=F) # shrink the size of submission
target_df <- target[-trainIndex,]
MulLogLoss(target_df,data.matrix(pred_ensemble))

pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))
write.csv(pred_ensemble,file='../submission_max_047.csv', quote=FALSE,row.names=FALSE)

# 0.511555 c(100,100)
# 0.5188986 hidden=c(50,50,50)
# 0.5203721 | 2.189166

# 0.5186584 gbm
# 0.5547743 rf mtries=30
# 0.5400635 rf mtries=22, ntree=2000, depth=80
# 0.536 rf mtries=30, ntree=2000, depth=50

# 1.462512 deepLearning
# 0.9236796 hidden=c(512,512),rate_decay=0.3,input_dropout_ratio = 0.2,hidden_dropout_ratios = c(0.5,0.5),l1=3e-6,l2=6e-6,max_w2=4
# 0.8025674 rate=0.1