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

levels(as.factor(train$target))
dummies <- dummyVars(~target, data = train)
target <- predict(dummies, newdata = train)
train$target <- NULL
train <- cbind(train, target)
head(train);colnames(train)

fitControl <- trainControl(method = "adaptive_cv", number = 10, repeats = 2, classProbs = TRUE,
                        summaryFunction = twoClassSummary, adaptive = list(min = 10,alpha = 0.05,method = 'BT',complete = TRUE))
gbmGrid <-  expand.grid(mtry=17)
fit <- train(x = train[,c(2:94)], y = as.factor(train[,95]), method = "rf",metric = 'ROC',trControl = fitControl,tuneLength = 8) #95:103
# tuneGrid = gbmGrid

library('e1071')
model <- svm(target~., train)
res <- predict(model, newdata=train)