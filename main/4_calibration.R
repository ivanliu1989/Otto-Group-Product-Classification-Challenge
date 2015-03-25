setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(data.table);require(caret)

ensemble <- data.matrix(fread('../submission_max_53.csv',stringsAsFactors = F))
a <- apply(ensemble[,2:10],1,sum)

for (i in 1:nrow(ensemble)){
    ensemble[i,2:10] <- ensemble[i,2:10] * 1 / a[i]
}
b <- apply(ensemble[,2:10],1,sum)

which(ensemble[,2:10] == 1)

ensemble = data.frame(ensemble)
write.csv(ensemble,file=paste0('../submission_max_53_cali.csv'), quote=FALSE,row.names=FALSE)
