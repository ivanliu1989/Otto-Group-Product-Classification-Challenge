# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(data.table)

ensem_prob <- matrix(0, nrow = 144368, ncol = 10, dimnames = list(NULL, NULL))
datadirectory <- '../otto-result' # 'results/best'
files <- list.files(datadirectory,full.names = T)

for (file in files){
    result <- data.frame(fread(file,header = T, stringsAsFactor = F))
    for (i in 1:10){
        for (j in 1:nrow(ensem_prob)){
            ensem_prob[j,i] <- max(result[j,i],ensem_prob[j,i])
        }
    }    
}

ensem_prob = format(ensem_prob, digits=2,scientific=F) # shrink the size of submission
ensem_prob = data.frame(1:nrow(ensem_prob),ensem_prob[,-1])
names(ensem_prob) = c('id', paste0('Class_',1:9))
head(ensem_prob)
write.csv(ensem_prob,file='../ensemble_xgboost_rf.csv', quote=FALSE,row.names=FALSE)

#######
result1 <- data.frame(fread('../benchmark_svm.csv',header = T, stringsAsFactor = F))
result2 <- data.frame(fread('../submission_max_17.csv',header = T, stringsAsFactor = F))

result <- result1
result[,2:10] <- result1[,2:10]*0.3 + result2[,2:10]*0.7
write.csv(result,file='../ensemble_xgboost_svm.csv', quote=FALSE,row.names=FALSE)
