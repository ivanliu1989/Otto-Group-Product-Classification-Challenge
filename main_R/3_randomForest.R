setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(data.table);require(caret);require(randomForest)
source('main_R/2_logloss_func.R')
train <- data.frame(fread('../train.csv', header=T, stringsAsFactor = F))
test <- data.frame(fread('../test.csv', header=T, stringsAsFactor = F))

library(doMC)
registerDoMC(cores = 4)

dim(train);set.seed(888)
trainIndex <- createDataPartition(train$target, p = .8,list = FALSE)
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