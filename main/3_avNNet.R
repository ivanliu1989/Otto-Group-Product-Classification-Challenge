setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
# setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(nnet)
source('main/2_logloss_func.R')
load(file='data/target.RData');load(file='data/raw_data_multi.RData')

dim(train);set.seed(888)
trainIndex <- createDataPartition(train[,95], p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]
# train_df <- train

fitControl <- trainControl(method = "none", number = 10, repeats = 5, classProbs = T, verbose = T)
gbmGrid <-  expand.grid(size=8, decay=0.3, maxit = 300)# bag=T)
fit <- train(x = train_df[,c(2:94)], y = as.factor(train_df[,95]), method ="nnet",# metric ='Kappa', 
             trControl = fitControl,do.trace=100, tuneGrid = gbmGrid,# repeats = 15, 
             trace=T)#, preProc = c("center","scale",'pca'))

val <- predict(fit, newdata=test_df[,-c(1,95)],type = "prob")
target_df <- target[-trainIndex,]
LogLoss(target_df,val)

### validation ###
varImpPlot(fit)
varUsed(fit)

### test ###
options(scipen=200)
res <- predict(fit, newdata=test,type = "prob")
submission <- data.table(cbind(id=test$id, res))
write.csv(submission,file='../first_try_rf.csv',row.names=F)

# 1.40 size=1, decay=0.1, bag=T, no pca
# 0.8665756 size=3, decay=0.1, bag=T, no pca
# 0.8574749  size=3, decay=0.1, repeats= 15, bag=T, no pca
# 0.8305999
# 0.8271034 size=4, decay=0.5, bag=T
# 0.8072769 size=4, decay=0.3, bag=T
# 0.7977291 size=4, decay=0.1, bag=T
# size=4, decay=0.15, bag=T

# 0.6609466 'nnet' size=7, decay=1