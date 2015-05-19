setwd('/Users/ivanliu/Google Drive/otto/Otto-Group-Product-Classification-Challenge');
rm(list=ls());gc()
require(data.table)

ensem_prob <- matrix(0, nrow = 144368, ncol = 10, dimnames = list(NULL, NULL))
ensem_prob <- matrix(0, nrow = 12378, ncol = 10, dimnames = list(NULL, NULL))
datadirectory <- '../Team_xgb/Val' # 'results/best'
files <- list.files(datadirectory,full.names = T)

for (file in files[1:15]){
    result <- data.frame(fread(file,header = T, stringsAsFactor = F))
    for (i in 1:10){
        for (j in 1:nrow(ensem_prob)){
            ensem_prob[j,i] <- sum(result[j,i],ensem_prob[j,i])
        }
    }    
}
# ensem_prob[,2:10] <- prop.table(ensem_prob[,2:10], 1) 
ensem_prob[,2:10] <- ensem_prob[,2:10]/length(files[1:15])
MulLogLoss(target_df,ensem_prob[,2:10] )

ensem_prob = format(ensem_prob, digits=2,scientific=F) # shrink the size of submission
ensem_prob = data.frame(1:nrow(ensem_prob),ensem_prob[,-1])
names(ensem_prob) = c('id', paste0('Class_',1:9))
head(ensem_prob)
write.csv(ensem_prob,file='../ensemble_nnet.csv', quote=FALSE,row.names=FALSE)

#######
result1 <- data.frame(fread('../nnet41.csv',header = T, stringsAsFactor = F))
result2 <- data.frame(fread('../submission_max_10.csv',header = T, stringsAsFactor = F))

result <- result1
result[,2:10] <- result1[,2:10]*0.65 + result2[,2:10]*0.35
write.csv(result,file='../final_sub.csv', quote=FALSE,row.names=FALSE)
