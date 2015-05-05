setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
require(data.table)
rm(list=ls());gc()

train <- data.frame(fread('../train.csv', header=T, stringsAsFactor = F))
test <- data.frame(fread('../test.csv', header=T, stringsAsFactor = F))

train_interact <- NULL
train_interact <- as.data.frame(model.matrix(target~.^2-1,train[,-1]))
dim(train_interact)
train_interact <- cbind(id =train$id, train_interact, target = train$target)
train_interact$target
head(train_interact)
head(train)

test_interact <- NULL
test_interact <- as.data.frame(model.matrix(~.^2-1,test[,-1]))
dim(test_interact)
test_interact <- cbind(id =test$id, test_interact)
test_interact$id
head(test_interact)
head(test)

write.csv(train_interact,file='../train_interact.csv', quote=FALSE,row.names=FALSE)
write.csv(test_interact,file='../test_interact.csv', quote=FALSE,row.names=FALSE)

### New features ###
train$target2 <- 1
train$target3 <- 1
train[,97] <- train$target
train[,95] <- apply(train[,2:94],1,function(x) length(which(x==0)))
train[,96] <- apply(train[,2:94],1,sum)
names(train) <- c('id',paste0('feat_',1:93),'sumNonZero','sumValue','target')

test$target2 <- 1
test$target3 <- 1
test[,95] <- apply(test[,2:94],1,function(x) length(which(x==0)))
test[,96] <- apply(test[,2:94],1,sum)
names(test) <- c('id',paste0('feat_',1:93),'sumNonZero','sumValue')

write.csv(train,file='../train_new.csv',row.names=F)
write.csv(test,file='../test_new.csv',row.names=F)

### log transfer ###
head(train[,-which(names(train) %in% c("id","target"))])
train[,-which(names(train) %in% c("id","target"))] <- log(1+train[,-which(names(train) %in% c("id","target"))])

head(test[,-which(names(test) %in% c("id"))])
test[,-which(names(test) %in% c("id"))] <- log(1+test[,-which(names(test) %in% c("id"))])

head(train);head(test)

write.csv(train,file='../train_new_log.csv',row.names=F)
write.csv(test,file='../test_new_log.csv',row.names=F)

### scale ###
source(file='main_R/2_logloss_func.R')

all_df <- rbind(train[,-which(names(train) %in% c("id","target"))], test[,-which(names(test) %in% c("id"))])
dim(train);dim(test);dim(all_df)
# scale(train[,2])
all_df <- apply(all_df,2,rangeScale) 
all_df <- apply(all_df,2,center_scale) 

train[,-which(names(train) %in% c("id","target"))] <- all_df[1:nrow(train),]
test[,-which(names(test) %in% c("id"))] <- all_df[(nrow(train)+1):nrow(all_df),]
head(train)
head(test)

write.csv(train,file='../train_new_log_scale_center.csv',row.names=F)
write.csv(test,file='../test_new_log_scale_center.csv',row.names=F)
