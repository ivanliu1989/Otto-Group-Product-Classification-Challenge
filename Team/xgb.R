setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost);require(data.table)
source('main_R/2_logloss_func.R');load(file='data/target.RData');
load(file='data/raw_data_log.RData');
#train <- fread('../train.csv', header=T, stringsAsFactor = F,data.table=F)
#test <- fread('../test.csv', header=T, stringsAsFactor = F, data.table=F)
folds <- fread('data/train_folds.csv', header=T, stringsAsFactor = F, data.table=F)$test_fold
library(doMC);registerDoMC(cores = 3)
options(scipen=3)

train_tot <- cbind(train, target, folds)
train_tot <- shuffle(train_tot) #<<============#
trainIndex <- which(folds == 0)
target_test <- train_tot[trainIndex,'target'];target_train <- train_tot[-trainIndex,'target']
train_test <- train_tot[trainIndex,2:94];train_train <- train_tot[-trainIndex,2:94]
target_df <- train_tot[trainIndex,96:104]

test = test[,-which(names(test) %in% c("id"))] #test

y = target_train
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)
y2 = target_test
y2 = gsub('Class_','',y2)
y2 = as.integer(y2)-1 #xgboost take features in [0,numOfClass)
y_tot <- c(y,y2)

train_train <- as.matrix(train_train)
dtrain = matrix(as.numeric(train_train),nrow(train_train),ncol(train_train))
train_test <- as.matrix(train_test)
dtrain_test = matrix(as.numeric(train_test),nrow(train_test),ncol(train_test))
test <- as.matrix(test)
dtest = matrix(as.numeric(test),nrow(test),ncol(test))
train_tot <- as.matrix(train_tot[,2:94])
dtrain_tot = matrix(as.numeric(train_tot),nrow(train_tot),ncol(train_tot))

for (i in 6:30){
    seeds <- 9*i
    set.seed(seeds) #<<============#
    param <- list("objective" = "multi:softprob",
                  "eval_metric" = "mlogloss", 
                  "nthread" = 3, set.seed = seeds, eta=0.05, gamma = 0.01, #<<============#
                  "num_class" = 9, max.depth=8, min_child_weight=5,
                  subsample=0.7, colsample_bytree = 0.6)
    #0.05, 0.8, 0.9 | 0.01, 0.7, 0.6
    cv.nround = 668
    # 698
    
    ### Train the model ###
    bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)
    
    ### Pred Validation ###
    pred = predict(bst,dtrain_test)#, ntreelimit=1
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    score <- MulLogLoss(target_df,pred)
    pred = format(pred, digits=2,scientific=F) # shrink the size of submission
    pred = data.frame(trainIndex,pred)
    names(pred) = c('id', paste0('Class_',1:9))
    write.csv(pred,file=paste0('../Team_xgb/Val/valPred_Ivan_m',i,'_CV',score,'_xgb.csv'), 
              quote=FALSE,row.names=FALSE)
    print(paste0('Validation:',score,' Complete!'))
    ### Train the model ###
    bst = xgboost(param=param, data = dtrain_tot, label = y_tot, nround = cv.nround)
    
    ### Make prediction ###
    pred = predict(bst,dtest)#, ntreelimit=1
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    
    pred = format(pred, digits=2,scientific=F) # shrink the size of submission
    pred = data.frame(1:nrow(pred),pred)
    names(pred) = c('id', paste0('Class_',1:9))
    write.csv(pred,file=paste0('../Team_xgb/Pred/testPred_Ivan_m',i,'_xgb_',seeds,'.csv'), 
              quote=FALSE,row.names=FALSE)
    print(paste0('Model:',i,' Complete!'))
}

