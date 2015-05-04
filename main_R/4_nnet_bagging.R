setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(caret);require(methods);require(xgboost)

fit1 <- read.csv('../lasagne_nnet_0.433.csv')
fit2 <- read.csv('../lasagne_nnet_0.465965.csv')
exist <- read.csv('../submission_max_17.csv')

head(fit1);dim(fit1)
head(fit2);dim(fit2)
head(exist);dim(exist)
