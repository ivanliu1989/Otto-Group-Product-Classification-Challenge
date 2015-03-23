setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
# setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(methods);require(xgboost)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_multi.RData')
# load(file='data/raw_data_PCA.RData')

dim(train);set.seed(888)
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
dtest <- x[teind,]

### Set necessary parameter ###
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss", 
              "nthread" = 2, seed = 8,
              "num_class" = 9, max.depth=7, eta=0.03,
              gamma = 0.01, subsample=0.8, colsample_bytree = 0.9)
# reg:logistic | logloss | lambda = 0 (L2) | alpha = 0 (L1) | lambda_bias = 0  

# Run Cross Valication
cv.nround = 1000
bst.cv = xgb.cv(param=param, data = dtrain, label = y, nfold = 10, 
                nrounds=cv.nround,prediction = TRUE)
# bst.cv$dt
# pred <- bst.cv$pred

### Train the model ###
set.seed(88)
bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)

### Make prediction ###
pred = predict(bst,dtest)#, ntreelimit=1
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

### Ensemble ###
# pred5 <- pred
# pred_ensemble <- (pred1 + pred2 + pred3 + pred4 + pred5)/5
# for (i in 1:9){
#     for (j in 1:nrow(pred1)){
#         pred_ensemble[j,i] <- max(pred1[j,i],pred2[j,i],pred3[j,i],pred4[j,i],pred5[j,i])
#     }
# }

### Beta varialble ###
# pred = predict(bst,dtest)
# pred = matrix(pred,9,length(pred)/9)
# pred = t(pred)
# dtest <- cbind(dtest, pred)
# 
# pred = predict(bst,dtrain)
# pred = matrix(pred,9,length(pred)/9)
# pred = t(pred)
# dtrain <- cbind(dtrain,pred)

### Validation ###
target_df <- target[-trainIndex,]
MulLogLoss(target_df,pred)

### Output submission ###
pred_ensemble = format(pred_ensemble, digits=2,scientific=F) # shrink the size of submission
pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))
write.csv(pred_ensemble,file='submission_max_047.csv', quote=FALSE,row.names=FALSE)


# 0.4762001 pred3
# 0.4751727 /5
# 0.439 /max

# 0.4813311 max.depth = 6, eta = 0.1, nround = 400, gamma = 0.1, subsample=0.8
# 0.4759731 max.depth = 8, eta = 0.05, nround = 500, gamma = 0.05, subsample=0.8
# 0.4763 max.depth = 6, eta = 0.03, nround = 1500, gamma = 0.03, subsample=0.8
# 0.4760015 max.depth = 8, eta = 0.03, nround = 900, gamma = 0.03, subsample=0.8
