setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
# setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(nnet)
source('main/2_logloss_func.R')
load(file='data/target.RData');load(file='data/raw_data_multi.RData')

dim(train);set.seed(888);train <- shuffle(train)
trainIndex <- createDataPartition(train[,95], p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]
# train_df <- train

fit <- nnet(x = train_df[,c(2:94)], y = as.factor(train_df[,95]), weights=1, size=7, entropy=F, softmax=T,censored=F, skip=F, decay=0,maxit=100,abstol=1.0e-4,reltol=1.0e-8)
# linout, entropy, softmax, censored
# rang=1, Hess=T, MaxNWts=1000,

val <- predict(fit, newdata=test_df[,-c(1,95)],type = "class")
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


# 0.6609466 'nnet' size=7, decay=1