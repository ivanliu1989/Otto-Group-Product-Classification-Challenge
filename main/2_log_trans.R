setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
load(file='data/raw_data_multi.RData')

head(train[,-which(names(train) %in% c("id","target"))])
train[,-which(names(train) %in% c("id","target"))] <- log(2+train[,-which(names(train) %in% c("id","target"))])

head(test[,-which(names(test) %in% c("id"))])
test[,-which(names(test) %in% c("id"))] <- log(2+test[,-which(names(test) %in% c("id"))])

head(train);head(test)

save(train,test,file='data/raw_data_log.RData')
