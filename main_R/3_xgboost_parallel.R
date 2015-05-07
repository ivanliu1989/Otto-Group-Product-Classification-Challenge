setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost);require(data.table)
source('main_R/2_logloss_func.R');load(file='data/target.RData');
train <- fread('../train.csv', header=T, stringsAsFactor = F,data.table=F)
test <- fread('../test.csv', header=T, stringsAsFactor = F, data.table=F)
library(doMC)
registerDoMC(cores = 3)

# train_tot <- cbind(train, target)
# train_tot <- shuffle(train_tot) #<<============#
train <- train_tot[,1:95]; target <- train_tot[,96:104]
train = train[,-which(names(train) %in% c("id"))] #train
test = test[,-which(names(test) %in% c("id"))] #test

y = train[,'target']
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)
train <- as.matrix(train[,-94])
dtrain = matrix(as.numeric(train),nrow(train),ncol(train))
test <- as.matrix(test[,-1])
dtest = matrix(as.numeric(test),nrow(test),ncol(test))

for (i in 1:30){
    seeds <- 8 * i
    set.seed(seeds) #<<============#
    param <- list("objective" = "multi:softprob",
                  "eval_metric" = "mlogloss", 
                  "nthread" = 3, set.seed = seeds, eta=0.05, gamma = 0.01, #0.05
                  "num_class" = 9, max.depth=8, min_child_weight=5, 
                  subsample=0.7, colsample_bytree = 0.6) #0.8
    cv.nround = 698
    
    ### Train the model ###
    bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)
    
    ### Make prediction ###
    pred = predict(bst,dtest)#, ntreelimit=1
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    
    pred = format(pred, digits=2,scientific=F) # shrink the size of submission
    pred = data.frame(1:nrow(pred),pred)
    names(pred) = c('id', paste0('Class_',1:9))
    write.csv(pred,file=paste0('../xgboost/xgboost_', i, '.csv'), quote=FALSE,row.names=FALSE)
    print(paste0('Model:',i,' Complete!'))
}

