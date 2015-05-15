setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost);require(data.table)
source('main_R/2_logloss_func.R');load(file='data/target.RData');load(file='data/raw_data_log.RData');
# train <- fread('../train.csv', header=T, stringsAsFactor = F,data.table=F)
# test <- fread('../test.csv', header=T, stringsAsFactor = F, data.table=F)
library(doMC)
registerDoMC(cores = 3)

trainIndex <- createDataPartition(train$target, p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]

train_df <- shuffle(train_df) #<<============#
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

### Set necessary parameter ###
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss", 
              "nthread" = 3, set.seed = 8, eta=0.05, gamma = 0.05, #<<============#
              "num_class" = 9, max.depth=8, min_child_weight=5,
              subsample=0.6, colsample_bytree = 0.7)
#2025 | 0.02
# param <- list("objective" = "multi:softprob",
#               "eval_metric" = "mlogloss", 
#               "nthread" = 3, set.seed = 168, eta=0.05, gamma = 0.03, #0.05
#               "num_class" = 9, max.depth=6, min_child_weight=4, 
#               subsample=0.7, colsample_bytree = 0.6) #0.8
# max.depth = 8, eta = 0.05, nround = 668, gamma = 0.05, subsample=0.8, colsample_bytree = 0.9
# reg:logistic | logloss | lambda = 0 (L2) | alpha = 0 (L1) | lambda_bias = 0  

### Run Cross Valication
cv.nround = 741
# bst.cv = xgb.cv(param=param, data = dtrain, label = y, nfold = 5, 
#                 nrounds=cv.nround,prediction = TRUE)

### Train the model ###
set.seed(168) #<<============#
bst = xgboost(param=param, data = dtrain, label = y, nround = cv.nround)

### Make prediction ###
pred = predict(bst,dtest)#, ntreelimit=1
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

### Ensemble ###
# pred17 <- pred #<<============#
# save(pred17, file='../xgboost_pred17.RData') #<<============#
# pred_ensemble <- (pred1 + pred2 + pred3 + pred4 + pred5 + pred6 + pred7 + pred8 + pred9 + pred10)/10
# for (i in 1:9){
#     for (j in 1:nrow(pred1)){
#         pred_ensemble[j,i] <- max(pred1[j,i],pred2[j,i],pred3[j,i],pred4[j,i],pred5[j,i],pred6[j,i]
#                                   ,pred7[j,i],pred8[j,i],pred9[j,i],pred10[j,i],pred11[j,i],pred12[j,i],pred13[j,i],pred14[j,i],
#                                   pred15[j,i],pred16[j,i],pred17[j,i])
#     }
# }

### Validation ###
target_df <- target[-trainIndex,]
MulLogLoss(target_df,pred)

### Output submission ###
pred_ensemble = format(pred_ensemble, digits=2,scientific=F) # shrink the size of submission
pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))
write.csv(pred_ensemble,file='../xgboost/xgboost_1.csv', quote=FALSE,row.names=FALSE)


# 0.4762001 pred3
# 0.4751727 /5
# 0.439 /max

# 0.4759731 max.depth = 8, eta = 0.05, nround = 500, gamma = 0.05, subsample=0.8
# 0.4760015 max.depth = 8, eta = 0.03, nround = 900, gamma = 0.03, subsample=0.8

# 0.4709471 max.depth = 8, eta = 0.05, nround = 668, gamma = 0.05, subsample=0.8, colsample_bytree = 0.9
# 0.4693221 same

# 0.4699977 min_child_weight = 4
# 0.4764768 1
# 0.4685496 min=4, 0.8, 0.9, new_feat

# 0.4673917 avg 2
# 0.4376464 max 2