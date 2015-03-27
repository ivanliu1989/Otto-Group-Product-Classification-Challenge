setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret);require(nnet);#require(deepnet)
source('main_R/2_logloss_func.R')
load(file='data/target.RData');load(file='data/raw_data_log_scale.RData')

dim(train);set.seed(888);
trainIndex <- createDataPartition(train$target, p = .7,list = FALSE)
train_df <- train[trainIndex,];test_df  <- train[-trainIndex,]

train = train_df[,-which(names(train_df) %in% c("id"))] #train
test = test_df[,-which(names(test_df) %in% c("id"))] #test
train <- shuffle(train)

dummies <- dummyVars(~target, data = train)
y <- predict(dummies, newdata = train)

x = rbind(train[,-which(names(train) %in% c("target"))],test[,-which(names(test) %in% c("target"))])#[,-which(names(test) %in% c("target"))])
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:nrow(y)
teind = (nrow(train)+1):nrow(x)
dtrain <- x[trind,]
dtest <- x[teind,]

best <- 10
set.seed(888)
for (n in c(3,10,30,100,150,200,300,500)){
    fit <- nnet(y=y, x=dtrain, size=n, softmax=T, skip=T, decay=0.2, maxit=100, abstol=1.0e-4, reltol=1.0e-8, rang=1, MaxNWts=150000)
    # linout, entropy, softmax, censored
    # rang=1, Hess=T,weights=1, 
    val <- predict(fit, newdata=dtest,type = "raw")
    target_df <- target[-trainIndex,]
    b <- MulLogLoss(target_df,val)
    s <- ifelse(b<best,"(*)","")
    best <- ifelse(b<best,b,best)
    print(paste0("parameter: ",n," | Score: ",b,s))
}
# decay c(0.2,0.4,0.6,0.8,1) | 0.2
# size c(3,10,30,100,150,200,300,500)
# maxit c(100,300,1000)

### validation ###
varImpPlot(fit)
varUsed(fit)

### test ###
options(scipen=200)
res <- predict(fit, newdata=test,type = "prob")
submission <- data.table(cbind(id=test$id, res))
write.csv(submission,file='../first_try_rf.csv',row.names=F)


# 0.6233329 size=7, decay=0.5, maxit=100
# 0.594525886065025 size=7, decay=0.2, maxit=100, skip=T