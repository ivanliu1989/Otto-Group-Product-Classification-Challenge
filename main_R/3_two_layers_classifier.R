setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(methods);require(xgboost)
source('main_R/2_logloss_func.R');load(file='data/target.RData');load(file='data/raw_data_multi.RData')

trainIndex <- createDataPartition(train$target, p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]
metaIndex <- createDataPartition(train_df$target, p = .7,list = FALSE)
base_df <- train_df[metaIndex,];meta_df  <- train_df[-metaIndex,]
dim(base_df);dim(meta_df);dim(test_df);dim(train)

base_df <- shuffle(base_df)
meta_df <- shuffle(meta_df)
base_df = base_df[,-which(names(base_df) %in% c("id"))]
meta_df = meta_df[,-which(names(meta_df) %in% c("id"))]
test_df = test_df[,-which(names(test_df) %in% c("id"))]

base_y = as.integer(gsub('Class_','',base_df[,'target']))-1 
meta_y = as.integer(gsub('Class_','',meta_df[,'target']))-1 
test_y = as.integer(gsub('Class_','',test_df[,'target']))-1 

base_x = as.matrix(base_df[,-which(names(base_df) %in% c("target"))])
base_x = matrix(as.numeric(base_x),nrow(base_x),ncol(base_x))
meta_x = as.matrix(meta_df[,-which(names(meta_df) %in% c("target"))])
meta_x = matrix(as.numeric(meta_x),nrow(meta_x),ncol(meta_x))
test_x = as.matrix(test_df[,-which(names(test_df) %in% c("target"))])
test_x = matrix(as.numeric(test_x),nrow(test_x),ncol(test_x))

### train meta ###
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss", 
              "nthread" = 2, set.seed = 168, eta=0.1, gamma = 0.1,
              "num_class" = 9, max.depth=8, min_child_weight=4,
              subsample=1, colsample_bytree = 1)
cv.nround = 100
bst = xgboost(param=param, data = base_x, label = base_y, nround = cv.nround)

meta_pred = predict(bst,meta_x)
meta_pred = matrix(meta_pred,9,length(meta_pred)/9)
meta_pred = t(meta_pred)

dummies <- dummyVars(~target, data = meta_df)
meta_target <- predict(dummies, newdata = meta_df)

MulLogLoss(meta_target,meta_pred)

### train meta+base ###
test_pred = predict(bst,test_x)
test_pred = matrix(test_pred,9,length(test_pred)/9)
test_pred = t(test_pred)

# meta_pred <- cbind(meta_x,meta_pred)
# test_pred <- cbind(test_x,test_pred)
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss", 
              "nthread" = 2, set.seed = 168, eta=0.05, gamma = 0.05,
              "num_class" = 9, max.depth=8, min_child_weight=4,
              subsample=1, colsample_bytree = 1)
cv.nround = 668

bst_meta = xgboost(param=param, data = meta_pred, label = meta_y, nround = cv.nround)

test_p = predict(bst_meta,test_pred)
test_p = matrix(test_p,9,length(test_p)/9)
test_p = t(test_p)

dummies_t <- dummyVars(~target, data = test_df)
test_target <- predict(dummies_t, newdata = test_df)

MulLogLoss(test_target,test_p)

