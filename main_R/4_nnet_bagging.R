setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost);require(data.table)

fit1 <- data.frame(fread('../lasagne_nnet_rect_0.423485.csv', header=T, stringsAsFactor = F))
fit2 <- data.frame(fread('../lasagne_nnet_pca_0.468737.csv', header=T, stringsAsFactor = F))
fit3 <- data.frame(fread('../lasagne_nnet_0.477.csv', header=T, stringsAsFactor = F))
fit4 <- data.frame(fread('../lasagne_nnet_pca_0.478259.csv', header=T, stringsAsFactor = F))
fit5 <- data.frame(fread('../lasagne_nnet_pca_0.454.csv', header=T, stringsAsFactor = F))

head(fit1);dim(fit1)
head(fit2);dim(fit2)
head(fit3);dim(fit3)
head(fit4);dim(fit4)
head(fit5);dim(fit5)

exist <- data.frame(fread('../nnet_pca_xgb_ensemble_0.419.csv', header=T, stringsAsFactor = F))
head(fit1);dim(fit1)
head(exist);dim(exist)

bagging <- exist
bagging[,2:10] <- (fit2[,2:10] + fit4[,2:10] + fit5[,2:10])/3
head(bagging);dim(bagging)

write.csv(bagging,file='nnet_pca_bagging_3.csv', quote=FALSE,row.names=FALSE)

exist <- data.matrix(ensem_prob)
class(exist)

exist[,2:10] <- prop.table(exist[,2:10], 1) 

head(exist)
exist <- as.data.frame(exist)

write.csv(exist,file='../bagging_nnet_14_rowsum.csv', quote=FALSE,row.names=FALSE)

nnet <- data.frame(fread('../results/bagging_nnet_14_rowsum.csv', header=T, stringsAsFactor = F))
xgb <- data.frame(fread('../results/submission_max_17_rowsum.csv', header=T, stringsAsFactor = F))
submission <- fread('../results/nnet_pca_xgb_ensemble_2.csv', header=T, stringsAsFactor=F, data.table=F)      
submission[,2:10] <- (xgb[,2:10] + nnet[,2:10])/2
write.csv(nnet,file='../nnet_pca_xgb_ensemble_3.csv', quote=FALSE,row.names=FALSE)

###############
### Bagging ###
###############
ensem_prob <- matrix(0, nrow = 144368, ncol = 10, dimnames = list(NULL, NULL))
datadirectory <- '../py_nnet' # 'results/best'
files <- list.files(datadirectory,full.names = T)

for (file in files){
    result <- data.frame(fread(file,header = T, stringsAsFactor = F))
    for (i in 1:10){
        for (j in 1:nrow(ensem_prob)){
            ensem_prob[j,i] <- sum(result[j,i],ensem_prob[j,i])
        }
    }    
}
ensem_prob[,2:10] <- ensem_prob[,2:10] / length(files)

ensem_prob = format(ensem_prob, digits=2,scientific=F) # shrink the size of submission
ensem_prob = data.frame(1:nrow(ensem_prob),ensem_prob[,-1])
names(ensem_prob) = c('id', paste0('Class_',1:9))
head(ensem_prob)
write.csv(ensem_prob,file='../bagging_nnet_14_mean.csv', quote=FALSE,row.names=FALSE)

exist <- data.matrix(ensem_prob)
class(exist)
exist[,2:10] <- prop.table(exist[,2:10], 1) 
head(exist)
exist <- as.data.frame(exist)
write.csv(exist,file='../bagging_nnet_14_rowsum.csv', quote=FALSE,row.names=FALSE)

fit2 <- data.frame(fread('../nnet_pca_xgb_ensemble_0.419.csv', header=T, stringsAsFactor = F))
head(fit2)
