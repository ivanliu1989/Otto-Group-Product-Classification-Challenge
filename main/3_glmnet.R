# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(glmnet)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_multi.RData')
table(train$target)

dim(train);set.seed(888)
trainIndex <- createDataPartition(train[,95], p = .7,list = FALSE)
train_df <- train[trainIndex,]
test_df  <- train[-trainIndex,]
# train_df <- train

fit <- glmnet(as.factor(target) ~ ., data=train_df[,-1], family="multinomial",alpha=0.5,standardize=T,
              type.logistic="Newton",type.multinomial="grouped")
#family="mgaussian" , #alpha=1 is the lasso penalty, and alpha=0 the ridge penalty
val <- predict(fit, newdata=test_df,type = "prob")
target_df <- target[-trainIndex,]
LogLoss(target_df,val)

### validation ###
varImpPlot(fit)
varUsed(fit)

### test ###
options(scipen=200)
res <- predict(fit, newdata=test,type = "prob")
eps=10^-15
res[res < eps] <- eps;
res[res > 1 - eps] <- 1 - eps;
res[which(res[,2]==1),]
submission <- data.table(cbind(id=test$id, res))
write.csv(submission,file='../first_try_rf.csv',row.names=F)

# 0.5963169 : ntree=250, mtry=10
# 0.623675451306547: ntree=100, mtry=10 | 