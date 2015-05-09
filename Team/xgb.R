setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost);require(data.table)
source('main_R/2_logloss_func.R');load(file='data/target.RData');
load(file='data/raw_data_log.RData');
# train <- fread('../train.csv', header=T, stringsAsFactor = F,data.table=F)
# test <- fread('../test.csv', header=T, stringsAsFactor = F, data.table=F)
folds <- fread('data/train_folds.csv', header=T, stringsAsFactor = F, data.table=F)$test_fold
library(doMC)
registerDoMC(cores = 3)

trainIndex <- which(folds == 0)
target_df <- target[trainIndex,];target_train <- target[-trainIndex,]
train_test <- train[which(folds == 0),];train_train <- train[which(folds != 0),]
train_tot <- cbind(train_train, target_train)
train_tot <- shuffle(train_tot) #<<============#
train <- train_tot[,1:95]; target <- train_tot[,96:104]
train = train[,-which(names(train) %in% c("id"))] #train
train_test = train_test[,-which(names(train_test) %in% c("id"))] #train_test
test = test[,-which(names(test) %in% c("id"))] #test



y = train[,'target']
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)
train <- as.matrix(train[,-94])
dtrain = matrix(as.numeric(train),nrow(train),ncol(train))
train_test <- as.matrix(train_test[,-94])
dtrain_test = matrix(as.numeric(train_test),nrow(train_test),ncol(train_test))
test <- as.matrix(test)
dtest = matrix(as.numeric(test),nrow(test),ncol(test))

for (i in 1:30){
    seeds <- 9 * i
    set.seed(seeds) #<<============#
    param <- list("objective" = "multi:softprob",
                  "eval_metric" = "mlogloss", 
                  "nthread" = 3, set.seed = seeds, eta=0.05, gamma = 0.01, #0.05
                  "num_class" = 9, max.depth=8, min_child_weight=5, 
                  subsample=0.7, colsample_bytree = 0.6) #0.8
    cv.nround = 698
    
    ### Train the model ###
    bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)
    
    ### Pred Validation ###
    pred = predict(bst,train_test)#, ntreelimit=1
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    score <- MulLogLoss(target_df,pred)
    pred = format(pred, digits=2,scientific=F) # shrink the size of submission
    pred = data.frame(trainIndex,pred)
    names(pred) = c('id', paste0('Class_',1:9))
    write.csv(pred,file=paste0('../Team_xgb/Val/valPred_Ivan_m',i,'CV',score,'_xgb.csv'), 
              quote=FALSE,row.names=FALSE)
    
    ### Make prediction ###
    pred = predict(bst,dtest)#, ntreelimit=1
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    
    pred = format(pred, digits=2,scientific=F) # shrink the size of submission
    pred = data.frame(1:nrow(pred),pred)
    names(pred) = c('id', paste0('Class_',1:9))
    write.csv(pred,file=paste0('../Team_xgb/Pred/testPred_Ivan_m',i,'_xgb.csv'), 
              quote=FALSE,row.names=FALSE)
    print(paste0('Model:',i,' Complete!'))
}

