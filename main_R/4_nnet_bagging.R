setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost);require(data.table)

fit1 <- data.frame(fread('../lasagne_nnet_0.467835.csv', header=T, stringsAsFactor = F))
fit2 <- data.frame(fread('../lasagne_nnet_pca_0.468737.csv', header=T, stringsAsFactor = F))
fit3 <- data.frame(fread('../lasagne_nnet_0.477.csv', header=T, stringsAsFactor = F))
fit4 <- data.frame(fread('../lasagne_nnet_pca_0.478259.csv', header=T, stringsAsFactor = F))
fit5 <- data.frame(fread('../lasagne_nnet_pca_0.454.csv', header=T, stringsAsFactor = F))

head(fit1);dim(fit1)
head(fit2);dim(fit2)
head(fit3);dim(fit3)
head(fit4);dim(fit4)
head(fit5);dim(fit5)

exist <- data.frame(fread('../submission_max_17.csv', header=T, stringsAsFactor = F))
head(exist);dim(exist)

bagging <- exist
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

nnet <- data.frame(fread('../nnet_pca_bagging_ 0.43902_3.csv', header=T, stringsAsFactor = F))
xgb <- data.frame(fread('../submission_max_17_rowsum.csv', header=T, stringsAsFactor = F))

xgb[,2:10] <- (xgb[,2:10] + nnet[,2:10])/2
write.csv(xgb,file='nnet_pca_xgb_ensemble.csv', quote=FALSE,row.names=FALSE)
