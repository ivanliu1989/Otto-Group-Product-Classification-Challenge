setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret)
load(file='data/raw_data_multi.RData')
source(file='main_R/2_logloss_func.R')

### New features ###
train$sumNonZero <- apply(train[,2:94],1,function(x) length(which(x==0)))
test$sumNonZero <- apply(test[,2:94],1,function(x) length(which(x==0)))

### log transfer ###
head(train[,-which(names(train) %in% c("id","target"))])
train[,-which(names(train) %in% c("id","target"))] <- log(1+train[,-which(names(train) %in% c("id","target"))])

head(test[,-which(names(test) %in% c("id"))])
test[,-which(names(test) %in% c("id"))] <- log(1+test[,-which(names(test) %in% c("id"))])

head(train);head(test)

save(train,test,file='data/raw_data_log_newFeat.RData')
load('data/raw_data_log_newFeat.RData')

### scale ###
all_df <- rbind(train[,-which(names(train) %in% c("id","target"))], test[,-which(names(test) %in% c("id"))])
dim(train);dim(test);dim(all_df)
# scale(train[,2])
all_df <- apply(all_df,2,rangeScale) 
all_df <- apply(all_df,2,center_scale) 

train[,-which(names(train) %in% c("id","target"))] <- all_df[1:nrow(train),]
test[,-which(names(test) %in% c("id"))] <- all_df[(nrow(train)+1):nrow(all_df),]
head(train)
head(test)

save(train,test,file='data/raw_data_log_scale_range_new_feat.RData')

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
