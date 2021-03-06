setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(methods);require(data.table);library(h2o)
source('main_R/2_logloss_func.R');load(file='data/target.RData');
train <- fread('../train.csv', header=T, stringsAsFactor = F,data.table=F)
test <- fread('../test.csv', header=T, stringsAsFactor = F, data.table=F)
options(scipen=3)

train_tot <- cbind(train, target)
train_tot <- shuffle(train_tot) #<<============#
train <- train_tot[,1:95]; target <- train_tot[,96:104]
train = train[,-which(names(train) %in% c("id"))] #train
test = test[,-which(names(test) %in% c("id"))] #test

localH2O <- h2o.init(nthread=3,Xmx="12g")
h2o.shutdown(localH2O)

for(i in 1:93){
    train[,i] <- as.numeric(train[,i])
    train[,i] <- sqrt(train[,i]+(3/8))
}
for(i in 1:93){
    test[,i] <- as.numeric(test[,i])
    test[,i] <- sqrt(test[,i]+(3/8))
}

train.hex <- as.h2o(localH2O,train)
test.hex <- as.h2o(localH2O,test)

predictors <- 1:(ncol(train.hex)-1)
response <- ncol(train.hex)

for(i in 1:20){
print(i)
model <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=train.hex,
                          classification=T,
                          activation="RectifierWithDropout",
                          hidden=c(1024,512,256),
                          hidden_dropout_ratio=c(0.5,0.5,0.5),
                          input_dropout_ratio=0.05,
                          epochs=100,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=2000,
                          max_w2=10,
                          seed=8)

# model <- h2o.deeplearning(x=predictors,
#                           y=response,
#                           data=train.hex,
#                           classification=T,
#                           activation="RectifierWithDropout",
#                           hidden=c(800,500,300),
#                           hidden_dropout_ratio=c(0.25,0.25,0.25),
#                           input_dropout_ratio=0.15,
#                           epochs=80,
#                           l1=1e-5,
#                           l2=1e-5,
#                           rho=0.99,
#                           epsilon=1e-8,
#                           train_samples_per_iteration=2000,
#                           max_w2=10,
#                           seed=8)

# pred <- as.data.frame(h2o.predict(model,test.hex))[,2:10]
# score <- MulLogLoss(target_df,pred)
# pred = format(pred, digits=2,scientific=F)
# pred = data.frame(trainIndex,pred)
# names(pred) = c('id', paste0('Class_',1:9))
# write.csv(pred,file=paste0('../Team_h2o/Val/valPred_Ivan_m',i,'_CV',score,'_h2odl.csv'), 
#           quote=FALSE,row.names=FALSE)

pred = as.data.frame(h2o.predict(model,test.hex))[,2:10]
pred = format(pred, digits=2,scientific=F)
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file=paste0('../Team_h2o/testPred_Ivan_m',i,'_h2odl.csv'), 
          quote=FALSE,row.names=FALSE)
print(paste0('Model:',i,' Complete!'))

}      

                   

