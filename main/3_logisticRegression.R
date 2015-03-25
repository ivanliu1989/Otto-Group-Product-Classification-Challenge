setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(methods);require(xgboost)

tot_round <- 50
for (i in 1:tot_round){
    
    set.seed((9+i*8)) #<<============#
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
                  "nthread" = 2, set.seed = (9+i*8), eta=0.05, gamma = 0.05, #<<============#
                  "num_class" = 9, max.depth=8, min_child_weight=4,
                  subsample=0.8, colsample_bytree = runif(1,0.85,0.95))
    cv.nround = 668
    
    ### Train the model ###
    bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)
    
    ### Make prediction ###
    pred = predict(bst,dtest)#, ntreelimit=1
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    
    save(pred, file=paste0('../xgboost_pred',(53+i),'.RData')) #<<============#

}

### Ensemble ###
rm(list=ls());gc()
require(data.table)

datadirectory <- '../otto-result' # 'results/best'
files <- list.files(datadirectory,full.names = T)
i <- 36
for (file in files){
    i <- i+1
    load(file)    
    write.csv(pred,file=paste0('../sub_tree_',i,'.csv'), quote=FALSE,row.names=FALSE)
}

all_result <- list()
j=1
for (file in files){
    all_result[[j]] <- as.data.frame(fread(file,stringsAsFactors = F))
    j <- j + 1
}

pred_ensemble <- matrix(0, nrow = 144368, ncol = 9, dimnames = list(NULL, NULL))
for (i in 1:9){
    for (j in 1:nrow(pred_ensemble)){
        pred_ensemble[j,i] <- max(all_result[[1]][j,i],all_result[[2]][j,i],all_result[[3]][j,i],all_result[[4]][j,i],all_result[[5]][j,i],all_result[[6]][j,i],
                                  all_result[[7]][j,i],all_result[[8]][j,i],all_result[[9]][j,i],all_result[[10]][j,i],all_result[[11]][j,i],all_result[[12]][j,i],
                                  all_result[[13]][j,i],all_result[[14]][j,i],all_result[[15]][j,i],all_result[[16]][j,i],all_result[[17]][j,i],all_result[[18]][j,i],
                                  all_result[[19]][j,i],all_result[[20]][j,i],all_result[[21]][j,i],all_result[[22]][j,i],all_result[[23]][j,i],all_result[[24]][j,i],
                                  all_result[[25]][j,i],all_result[[26]][j,i],all_result[[27]][j,i],all_result[[28]][j,i],all_result[[29]][j,i],all_result[[30]][j,i],
                                  all_result[[31]][j,i],all_result[[32]][j,i],all_result[[33]][j,i],all_result[[34]][j,i],all_result[[35]][j,i],all_result[[36]][j,i],
                                  all_result[[37]][j,i],all_result[[38]][j,i],all_result[[39]][j,i],all_result[[40]][j,i],all_result[[41]][j,i],all_result[[42]][j,i],
                                  all_result[[43]][j,i],all_result[[44]][j,i],all_result[[45]][j,i],all_result[[46]][j,i],all_result[[47]][j,i],all_result[[48]][j,i],
                                  all_result[[49]][j,i],all_result[[50]][j,i],all_result[[51]][j,i],all_result[[52]][j,i],all_result[[53]][j,i])
    }
}

pred_ensemble = format(pred_ensemble, digits=2,scientific=F) # shrink the size of submission
pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))
write.csv(pred_ensemble,file=paste0('../submission_max_',length(files),'.csv'), quote=FALSE,row.names=FALSE)
