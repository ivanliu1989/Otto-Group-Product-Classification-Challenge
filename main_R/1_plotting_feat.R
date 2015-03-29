setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret)
load(file='data/raw_data_multi.RData')

boxplot(feat_1~target, data=train, notch=F)
boxplot(feat_2~target, data=train)
boxplot(feat_3~target, data=train)
boxplot(feat_4~target, data=train)
boxplot(sumNonZero~target, data=train)

plot(train$feat_1)

head(train)

train[,96] <- train$target
train[,95] <- apply(train[,2:94],1,function(x) length(which(x==0)))
names(train) <- c('id',paste0('feat_',1:93),'sumNonZero','target')

train$extra <- apply(train[,2:94],1,sum)
boxplot(extra~target, data=train)

#group <- kmeans(train[,2:94], centers=9,iter.max = 1000000,nstart = 27)
