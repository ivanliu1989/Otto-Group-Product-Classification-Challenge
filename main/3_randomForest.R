# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(randomForest)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_multi.RData')
table(train$target)

dim(train);set.seed(17)
trainIndex <- createDataPartition(train[,95], p = .7,list = FALSE)
train_df <- train[trainIndex,]
test_df  <- train[-trainIndex,]

fit <- randomForest(x = train_df[,c(2:94)], y = as.factor(train_df[,95]), data=train_df, importance=F, ntree=250)
val <- predict(fit, newdata=test_df,type = "prob")
target_df <- target[-trainIndex,]
LogLoss(target_df,val)

### validation ###
varImpPlot(fit)
varUsed(fit)

### test ###
res <- predict(fit, newdata=test,type = "prob")
submission <- cbind(id=test$id, res)
write.csv(submission,file='../first_try_rf.csv',row.names=F)