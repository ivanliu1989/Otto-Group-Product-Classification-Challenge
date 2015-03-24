setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret)
load(file='data/raw_data_multi.RData')

### log transfer ###
head(train[,-which(names(train) %in% c("id","target"))])
train[,-which(names(train) %in% c("id","target"))] <- log(2+train[,-which(names(train) %in% c("id","target"))])

head(test[,-which(names(test) %in% c("id"))])
test[,-which(names(test) %in% c("id"))] <- log(2+test[,-which(names(test) %in% c("id"))])

head(train);head(test)

save(train,test,file='data/raw_data_log.RData')


### scale ###
all_df <- rbind(train[,-which(names(train) %in% c("id","target"))], test[,-which(names(test) %in% c("id"))])
dim(train);dim(test);dim(all_df)
scaleFit <- preProcess(all_df, method = "scale")
train[,-which(names(train) %in% c("id","target"))] <- predict(scaleFit,train[,-which(names(train) %in% c("id","target"))])
test[,-which(names(test) %in% c("id"))] <- predict(scaleFit,test[,-which(names(test) %in% c("id"))])
head(train)
head(test)

save(train,test,file='data/raw_data_log_scale.RData')
