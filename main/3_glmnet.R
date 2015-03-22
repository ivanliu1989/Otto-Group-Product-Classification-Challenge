# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(glmnet)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_multi.RData')
# load(file='data/raw_data_PCA.RData')

dim(train);set.seed(888)
trainIndex <- createDataPartition(train$target, p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]

train = train_df[,-which(names(train_df) %in% c("id"))] #train
test = test_df[,-which(names(test_df) %in% c("id"))] #test

y = train[,'target']
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-which(names(train) %in% c("target"))],test[,-which(names(test) %in% c("target"))])#[,-which(names(test) %in% c("target"))])
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)
dtrain <- x[trind,]
dtest <- x[teind,]

# train_df <- train

fit <- glmnet(y=target[trainIndex,3], x=dtrain, family="binomial",alpha=0.7,standardize=F,
              type.logistic="Newton", nlambda=100, intercept=T, maxit=10^5,type.multinomial="ungrouped")
#family="mgaussian" , #alpha=1 is the lasso penalty, and alpha=0 the ridge penalty
# ungrouped,multinomial
val <- predict(fit, newx=dtest,type = "response")
target_df <- target[-trainIndex,3]
# MulLogLoss(target_df,val[,91])
LogLoss(target_df,val[,95])

### test ###
options(scipen=200)
res <- predict(fit, newdata=test,type = "prob")
submission <- data.table(cbind(id=test$id, res))
write.csv(submission,file='../first_try_rf.csv',row.names=F)

# 0.6414592 family="multinomial",alpha=0.5,standardize=T
# 0.6407393