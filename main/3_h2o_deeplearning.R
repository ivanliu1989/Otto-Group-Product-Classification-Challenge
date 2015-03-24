setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
library(h2o)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_multi.RData')

localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '8g')
h2o.clusterInfo(localH2O)
train <- as.h2o(localH2O, train, key="train")
test <- as.h2o(localH2O, test, key="test")

independent <- colnames(train[,2:94])
dependent <- "target"
fit <- h2o.gbm(y = dependent, x = independent, data = train, 
               n.trees = 150, interaction.depth = 5,
               n.minobsinnode = 2, shrinkage = 0.01, distribution= "multinomial")

fit <- h2o.deeplearning(y = dependent, x = independent, data = train, 
                        classification=T,activation="Tanh",
                        hidden=c(10,10,10),epochs=12,variable_importances=T,
                        override_with_best_model=T,nfolds=10,seed=8,loss='CrossEntropy',
                        epsilon=0.01,rate=0.1,rate_decay=0.1,nesterov_accelerated_gradient=T,
                        input_dropout_ratio=0.1,max_w2=4,l1=0.4,l2=0.4,shuffle_training_data=T)
#adaptive_rate=0.9,
pred <- h2o.predict(object = fit, newdata = test)

pred_ensemble = format(as.data.frame(pred[,2:10]), digits=2,scientific=F) # shrink the size of submission
pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))
write.csv(pred_ensemble,file='../submission_max_047.csv', quote=FALSE,row.names=FALSE)
