setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(methods);require(data.table);library(h2o)
source('main_R/2_logloss_func.R');load(file='data/target.RData');
train <- fread('../train.csv', header=T, stringsAsFactor = F,data.table=F)
test <- fread('../test.csv', header=T, stringsAsFactor = F, data.table=F)
folds <- fread('data/train_folds.csv', header=T, stringsAsFactor = F, data.table=F)$test_fold

options(scipen=3)

trainIndex <- which(folds == 0)
target_df <- target[trainIndex,];target_train <- target[-trainIndex,]
train_test <- train[which(folds == 0),];train_train <- train[which(folds != 0),]

head(train_train)
write.csv(train_train,file='../train_fold0.csv', quote=FALSE,row.names=FALSE)
write.csv(train_test,file='../test_fold0.csv', quote=FALSE,row.names=FALSE)
