# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(e1071);require(caret)
source('main_R/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_log.RData')
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

library(doMC)
registerDoMC(cores = 2)

fit <- svm(y=as.factor(y), x=dtrain, scale=T, type='C-classification', kernel='radial', degree=3, gamma=0.001,
           cost=1,cachesize=1024,tolerance=0.001,epsilon=0.1,shrinking=T,fitted=T,probability=T)
val <- predict(fit, newx=dtest,type = "prob")

target_df <- target[-trainIndex,]
MulLogLoss(target_df,val)

### test ###
options(scipen=200)
res <- predict(fit, newdata=test,type = "prob")
submission <- data.table(cbind(id=test$id, res))
write.csv(submission,file='../first_try_rf.csv',row.names=F)

# 