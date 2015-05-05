setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(data.table);library(doMC);require(e1071);require(caret)
registerDoMC(cores=4)
source('main_R/2_logloss_func.R')
load(file='data/target.RData')
train <- data.frame(fread('../train.csv', header=T, stringsAsFactor = F))
test <- data.frame(fread('../test.csv', header=T, stringsAsFactor = F))

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

fit <- svm(y=as.factor(y), x=dtrain, scale=T, type='C-classification', kernel='radial', degree=3, gamma=0.5,
           cost=4000,cachesize=1024,tolerance=0.001,epsilon=0.1,shrinking=T,fitted=T,probability=T)
# degree <- c(1:3)
# cost <- c(2^-5,2^-3,2^-1,2,2^3,2^5,2^7,2^9,2^11,2^13,2^15)
# gamma <- c(2^-15,2^-13,2^-11,2^-9,2^-7,2^-5,2^-3,2^-1,2,2^3)
pred <- predict(fit, dtest, probability=TRUE)
val <- attr(pred, "probabilities")

target_df <- target[-trainIndex,]
MulLogLoss(target_df,val)

### test ###
options(scipen=200)
res <- predict(fit, newdata=test,type = "prob")
submission <- data.table(cbind(id=test$id, res))
write.csv(submission,file='../first_try_rf.csv',row.names=F)
