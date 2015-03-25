setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(methods);require(xgboost)

tot_round <- 10
for (i in 1:tot_round){
    
    set.seed((8+i*8)) #<<============#
    source('main/2_logloss_func.R');load(file='data/target.RData');load(file='data/raw_data_multi.RData')
    
    train <- shuffle(train) #<<============#
    train = train[,-which(names(train) %in% c("id"))]
    test = test[,-which(names(test) %in% c("id"))]
    
    y = train[,'target']
    y = gsub('Class_','',y)
    y = as.integer(y)-1 
    
    x = rbind(train[,-which(names(train) %in% c("target"))],test)
    x = as.matrix(x)
    x = matrix(as.numeric(x),nrow(x),ncol(x))
    trind = 1:length(y)
    teind = (nrow(train)+1):nrow(x)
    dtrain <- x[trind,]
    dtest <- data.matrix(x[teind,])
    
    ### Set necessary parameter ###
    param <- list("objective" = "multi:softprob",
                  "eval_metric" = "mlogloss", 
                  "nthread" = 2, set.seed = (8+i*8), eta=0.05, gamma = 0.05, #<<============#
                  "num_class" = 9, max.depth=8, min_child_weight=1,
                  subsample=0.8, colsample_bytree = 0.9)
    cv.nround = 668
    
    ### Train the model ###
    bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)
    
    ### Make prediction ###
    pred = predict(bst,dtest)#, ntreelimit=1
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    
    save(pred, file=paste0('../xgboost_pred',(17+i),'.RData')) #<<============#

}