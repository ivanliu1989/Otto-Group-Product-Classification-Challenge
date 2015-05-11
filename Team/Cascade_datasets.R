setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost);require(data.table)
train <- fread('data/train_folds.csv', header=T, stringsAsFactor = F, data.table=F)

head(train)
train[which(train$target %in% paste0('Class_', c(1,6,7,8,9))),'group']<-1
train[which(train$target %in% paste0('Class_', c(2,3,4))),'group']<-2
train[which(train$target %in% paste0('Class_', 5)),'group']<-3
sum(table(train$group))
nrow(train)

write.csv(train,file='data/train_folds_cascade.csv', quote=FALSE,row.names=FALSE)
