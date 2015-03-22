# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(randomForest)
source('main/2_logloss_func.R')
load(file='data/target.RData');load(file='data/raw_data_multi.RData')

dim(train);set.seed(888)
trainIndex <- createDataPartition(train[,95], p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]
# train_df <- train

fit <- randomForest(as.factor(target) ~ ., data=train_df[,-1], importance=F, ntree=300,mtry=30)
val <- predict(fit, newdata=test_df,type = "prob")
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

# 0.5963169: ntree=250, mtry=10
# 0.630649626524021: ntree=250, mtry=5
# 0.623675451306547: ntree=100, mtry=10
# 0.582289597494042: ntree=1000, mtry=10
# 0.5757081 ntree=250,mtry=20