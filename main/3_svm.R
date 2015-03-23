# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_multi.RData')
# load(file='data/raw_data_PCA.RData')

dim(train);set.seed(888)
trainIndex <- createDataPartition(train$target, p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]

train = train_df[,-which(names(train_df) %in% c("id"))] #train
test = test_df[,-which(names(test_df) %in% c("id"))] #test

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
# train_df <- train

library(doMC)
registerDoMC(cores = 2)
mul_val <- target[-trainIndex,]
for (n in 1:9){
    #     n <- 1
    fitControl <- trainControl(method = "none", number = 10, repeats = 5, classProbs = T, verbose = T)
    gbmGrid <-  expand.grid(C=1)# bag=T)
    fit <- train(y=target[trainIndex,n], x=dtrain, method ="svmLinear",# metric ='Kappa', 
                 trControl = fitControl,do.trace=100, tuneGrid = gbmGrid,
                 trace=T, preProc = c("center","scale"), verbose=T)#,'pca'
    val <- predict(fit, newx=dtest,type = "prob")
    target_df <- target[-trainIndex,n]
    logloss <- LogLoss(target_df,val)
    print(paste0(logloss, '\n'))
    mul_val[,n] <- val
}
target_df <- target[-trainIndex,]
MulLogLoss(target_df,mul_val)


### test ###
options(scipen=200)
res <- predict(fit, newdata=test,type = "prob")
submission <- data.table(cbind(id=test$id, res))
write.csv(submission,file='../first_try_rf.csv',row.names=F)

# 