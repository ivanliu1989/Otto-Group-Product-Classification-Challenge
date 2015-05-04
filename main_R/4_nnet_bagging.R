setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost)

fit1 <- data.matrix(read.csv('../lasagne_nnet_0.467835.csv'))
fit2 <- data.matrix(read.csv('../lasagne_nnet_pca_0.468737.csv'))
fit3 <- data.matrix(read.csv('../lasagne_nnet_0.477.csv'))
fit4 <- data.matrix(read.csv('../lasagne_nnet_pca_0.478259.csv'))
fit5 <- data.matrix(read.csv('../lasagne_nnet_pca_0.454.csv'))

head(fit1);dim(fit1)
head(fit2);dim(fit2)
head(fit3);dim(fit3)
head(fit4);dim(fit4)
head(fit5);dim(fit5)

exist <- read.csv('../submission_max_17_rowsum.csv')
head(exist);dim(exist)

bagging <- fit2
bagging[,2:10] <- (fit2[,2:10] + fit4[,2:10] + fit5[,2:10])/3
head(bagging);dim(bagging)

write.csv(bagging,file='nnet_pca_bagging_3.csv', quote=FALSE,row.names=FALSE)

# exist <- data.matrix(exist)
# class(exist)
# 
# exist[,2:10] <- prop.table(exist[,2:10], 1) 
# 
# head(exist)
# exist <- as.data.frame(exist)
# 
# write.csv(exist,file='submission_max_17_rowsum.csv', quote=FALSE,row.names=FALSE)
