setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(methods);require(xgboost)

tot_round <- 50
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
                  "num_class" = 9, max.depth=8, min_child_weight=4,
                  subsample=0.8, colsample_bytree = runif(1,0.85,0.95))
    cv.nround = 668
    
    ### Train the model ###
    bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)
    
    ### Make prediction ###
    pred = predict(bst,dtest)#, ntreelimit=1
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    
    save(pred, file=paste0('../xgboost_pred',(17+i),'.RData')) #<<============#

}

### Ensemble ###
require(data.table)

pred_ensemble <- matrix(0, nrow = 144368, ncol = 9, dimnames = list(NULL, NULL))
datadirectory <- '../otto-result' # 'results/best'
files <- list.files(datadirectory,full.names = T)

for (file in files){
    load(file)    
}
ls()
for (i in 1:9){
    for (j in 1:nrow(pred_ensemble)){
        pred_ensemble[j,i] <- max(pred1[j,i],pred2[j,i],pred3[j,i],pred4[j,i],pred5[j,i],pred6[j,i]
                                  ,pred7[j,i],pred8[j,i],pred9[j,i],pred10[j,i])
    }
}
pred_ensemble = format(pred_ensemble, digits=2,scientific=F) # shrink the size of submission
pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))
write.csv(pred_ensemble,file=paste0('../submission_max_',length(files),'.csv'), quote=FALSE,row.names=FALSE)
