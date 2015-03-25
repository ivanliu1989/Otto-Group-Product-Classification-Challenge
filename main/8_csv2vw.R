setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
load(file='data/raw_data_log.RData')

### train ###
head(train,1)
train[,95]<-as.factor(train[,95])
levels(train[,95]) <- c(1:9)
col <- names(train)

row <- paste0(train[,95],paste0(" '",train[,1]),"|n ")
row2 <- sapply(2:94, function(i) {
    if(i==94){
        paste0(col[i],":",train[,i])
    }else{
        paste0(col[i],":",train[,i]," ")
    }
    })

train_vw <- cbind(row,row2)
train_vw_int <- matrix(data = "",nrow = nrow(train_vw),ncol = 1)
for(i in 1:dim(train_vw)[2]){
    train_vw_int[,1] <- paste0(train_vw_int[,1],train_vw[,i])
}

write.table(train_vw_int, "../train.vw", quote = F,row.names = F,col.names = F)


### test ###
head(test,1)
col <- names(test)

row <- paste0(1,paste0(" '",test[,1]),"|n ")
row2 <- sapply(2:94, function(i) {
    if(i==94){
        paste0(col[i],":",test[,i])
    }else{
        paste0(col[i],":",test[,i]," ")
    }
})

test_vw <- cbind(row,row2)
test_vw_int <- matrix(data = "",nrow = nrow(test_vw),ncol = 1)
for(i in 1:dim(test_vw)[2]){
    test_vw_int[,1] <- paste0(test_vw_int[,1],test_vw[,i])
}

write.table(test_vw_int, "../test.vw", quote = F,row.names = F,col.names = F)
