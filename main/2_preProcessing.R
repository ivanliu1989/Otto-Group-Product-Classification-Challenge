# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret)
load(file='data/raw_data_multi.RData')

### NearZeroVariance ###
nzv <- nearZeroVar(train, saveMetrics= TRUE)
nzv[nzv$nzv,]

nzv <- nearZeroVar(train)
train <- train[, -nzv]
dim(train)

### Correlated Predictors ###
descrCor <- cor(train[,2:81]) #94
summary(descrCor[upper.tri(descrCor)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
train <- train[,-highlyCorDescr]
descrCor2 <- cor(train[])
summary(descrCor2[upper.tri(descrCor2)])

### Linear Dependencies ###
comboInfo <- findLinearCombos(train[,2:81])
comboInfo
train[, -comboInfo$remove]

### NA ###
sum(is.na(train))
