setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(methods);require(xgboost)
source('main/2_logloss_func.R')
load(file='data/target.RData');load(file='data/raw_data_multi.RData')

dim(train);set.seed(888)
trainIndex <- createDataPartition(train[,95], p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]


train = train_df[,-1]
test = test_df[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test[,-ncol(test)])
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)
dtrain <- x[trind,]
dtest <- x[teind,]

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 4)

# Run Cross Valication
cv.nround = 50
bst.cv = xgb.cv(param=param, data = dtrain, label = y, 
                nfold = 10, nrounds=cv.nround)

# Train the model
set.seed(888)
bst = xgboost(param=param, data = dtrain, label = y, max.depth = 6, eta = 0.1, nround = 500, gamma = 0.1, subsample=0.8)

# Make prediction
pred = predict(bst,dtest)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)
pred3 <- pred
# pred_ensemble <- (pred1 + pred2 + pred3 + pred4 + pred5)/5

# Validation
target_df <- target[-trainIndex,]
LogLoss(target_df,pred3)

ptrain <- predict(bst, dtrain, outputmargin=TRUE)
ptest  <- predict(bst, dtest, outputmargin=TRUE)
setinfo(dtrain, "base_margin", ptrain)
setinfo(dtest, "base_margin", ptest)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)


# 0.4968521 max.depth=6, eta=0.3, nround=150, gamma=0.3, subsample=1
# 0.5220532 max.depth=6, eta=0.1, nround=150, gamma=0.3, subsample=1
# 0.498769 max.depth=6, eta=0.2, nround=150, gamma=0.3, subsample=1
# 0.5093105 max.depth=6, eta=0.4, nround=150, gamma=0.3, subsample=1
# 0.4993252 max.depth=6, eta=0.3, nround=150, gamma=0.1, subsample=1
# 0.4963108 max.depth=6, eta=0.3, nround=150, gamma=0.5, subsample=1
# 0.49975 max.depth=6, eta=0.3, nround=150, gamma=1, subsample=1
# 0.4985567 max.depth=7, eta=0.3, nround=150, gamma=0.5, subsample=1
# 0.5122933 max.depth=4, eta=0.3, nround=150, gamma=0.5, subsample=1
# 0.5038205 max.depth=5, eta=0.3, nround=150, gamma=0.5, subsample=1
# 0.4983714 max.depth = 4, eta = 0.1, nround = 1000, gamma = 0.1, subsample=1
# 0.4904834 max.depth = 6, eta = 0.1, nround = 500, gamma = 0.1, subsample=1
# 0.4843841 max.depth = 6, eta = 0.1, nround = 500, gamma = 0.1, subsample=0.7
# 0.4814787 max.depth = 6, eta = 0.1, nround = 500, gamma = 0.1, subsample=0.8
# 0.4762001 pred3