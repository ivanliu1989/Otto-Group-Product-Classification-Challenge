setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(data.table);require(caret)

pred_ensemble <- data.matrix(fread('../submission_sum_53.csv',stringsAsFactors = F))
pred_ensemble <- pred_ensemble[,2:10]
a <- apply(pred_ensemble,1,sum)

for (i in 1:nrow(pred_ensemble)){
    pred_ensemble[i,] <- pred_ensemble[i,] * 1 / a[i]
}
b <- apply(pred_ensemble,1,sum)

which(pred_ensemble == 1)

pred_ensemble = format(pred_ensemble, digits=2,scientific=F) # shrink the size of submission
pred_ensemble = data.frame(1:nrow(pred_ensemble),pred_ensemble)
names(pred_ensemble) = c('id', paste0('Class_',1:9))

head(pred_ensemble)
write.csv(pred_ensemble,file=paste0('../submission_sum_53_cali.csv'), quote=FALSE,row.names=FALSE)
