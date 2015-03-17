setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(data.table);require(caret)

# train <- as.data.frame(fread(unzip('data/train.csv.zip'),stringsAsFactors = F))
# test <- as.data.frame(fread(unzip('data/test.csv.zip'),stringsAsFactors = F))
# sampleSubmission <- fread('data/sampleSubmission.csv',stringsAsFactors = F)
# save(train,test,sampleSubmission, file='data/raw_data.RData')
load(file='data/raw_data.RData')

fitControl <- trainControl(method = "adaptive_cv", number = 10, repeats = 2, classProbs = TRUE,
                        summaryFunction = twoClassSummary, adaptive = list(min = 10,alpha = 0.05,method = 'BT',complete = TRUE))
fit <- train(target ~ ., data = train, method = "rf",metric = 'ROC',trControl = fitControl,tuneLength = 8)