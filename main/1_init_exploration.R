setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(data.table);require(caret)

train <- as.data.frame(fread(unzip('data/train.csv.zip'),stringsAsFactors = F))
test <- as.data.frame(fread(unzip('data/test.csv.zip'),stringsAsFactors = F))
sampleSubmission <- fread('data/sampleSubmission.csv',stringsAsFactors = F)
save(train,test, file='data/raw_data_multi.RData')
load(file='data/raw_data.RData')

levels(as.factor(train$target))
dummies <- dummyVars(~target, data = train)
target <- predict(dummies, newdata = train)
save(target, file='data/target.RData')
train$target <- NULL
train <- cbind(train, target)

