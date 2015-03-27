setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret)
load(file='data/raw_data_multi.RData')
source(file='main_R/2_logloss_func.R')

### log transfer ###
head(train[,-which(names(train) %in% c("id","target"))])
train[,-which(names(train) %in% c("id","target"))] <- log(1+train[,-which(names(train) %in% c("id","target"))])

head(test[,-which(names(test) %in% c("id"))])
test[,-which(names(test) %in% c("id"))] <- log(1+test[,-which(names(test) %in% c("id"))])

head(train);head(test)

save(train,test,file='data/raw_data_log_2.RData')
load('data/raw_data_log_2.RData')

### scale ###

all_df <- rbind(train[,-which(names(train) %in% c("id","target"))], test[,-which(names(test) %in% c("id"))])
dim(train);dim(test);dim(all_df)
# scale(train[,2])
train[,-which(names(train) %in% c("id","target"))] <- apply(train[,-which(names(train) %in% c("id","target"))],2,rangeScale) 
test[,-which(names(test) %in% c("id"))] <- apply(test[,-which(names(test) %in% c("id"))],2,rangeScale) 
head(train)
head(test)

save(train,test,file='data/raw_data_log_scale.RData')

### sparse matrix ###
id <- train$id
target <- as.factor(train$target)
train$id <- NULL;train$target <- NULL

train <- apply(train,2,as.factor)
levels(train[,1])
dummies <- dummyVars(~., data = train)
train_sparse <- predict(dummies, newdata = train)
train <- cbind(id,train_sparse,target)
colnames(train)
save(train,file='data/raw_data_sparse.RData')
