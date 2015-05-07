setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost);require(data.table)
source('main_R/2_logloss_func.R');load(file='data/target.RData');
train <- fread('../train.csv', header=T, stringsAsFactor = F,data.table=F)
test <- fread('../test.csv', header=T, stringsAsFactor = F, data.table=F)
library(doMC)
registerDoMC(cores = 4)

train_tot <- cbind(train, target)
train_tot <- shuffle(train_tot) #<<============#
train <- train_tot[,1:95]; target <- train_tot[,96:104]
trainIndex <- createDataPartition(train$target, p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]
train = train_df[,-which(names(train) %in% c("id"))] #train
test = test_df[,-which(names(test) %in% c("id"))] #test

y = train[,'target']
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-which(names(train) %in% c("target"))],test[,-which(names(test) %in% c("target"))])#[,-which(names(test) %in% c("target"))])
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)
dtrain <- x[trind,]
dtest <- data.matrix(x[teind,])

seeds <- 168
set.seed(seeds) #<<============#
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss", 
              "nthread" = 3, set.seed = seeds, eta=0.05, gamma = 0.2,
              "num_class" = 9, max.depth=8, min_child_weight=6,
              subsample=0.8, colsample_bytree = 0.9)
cv.nround = 698

### Train the model ###
bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)

### Make prediction ###
pred = predict(bst,dtest)#, ntreelimit=1
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

target_df <- target[-trainIndex,]
MulLogLoss(target_df,pred)

### Output submission ###
pred_ensemble = format(pred_ensemble, digits=2,scientific=F) # shrink the size of submission
pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))
write.csv(pred_ensemble,file='../xgboost/xgboost_1.csv', quote=FALSE,row.names=FALSE)
